"""RoboCasa adapter for RLDX's native ZeroMQ server (--use-sim-policy-wrapper).

Requires rldx in the caller's Python environment. Call start_episode once per
episode/subtask, observe after every env.step, and infer once per action chunk.
After a request timeout, close this client and create a new one.
"""

from collections import deque
from uuid import uuid4

import numpy as np

VIDEO_KEYS = (
    "video.robot0_agentview_left",
    "video.robot0_agentview_right",
    "video.robot0_eye_in_hand",
)
STATE_SIZES = {
    "state.gripper_qpos": 2,
    "state.base_position": 3,
    "state.base_rotation": 4,
    "state.end_effector_position_relative": 3,
    "state.end_effector_rotation_relative": 4,
}
ACTION_SIZES = {
    "action.end_effector_position": 3,
    "action.end_effector_rotation": 3,
    "action.gripper_close": 1,
    "action.base_motion": 4,
    "action.control_mode": 1,
}


class RLDXClient:
    def __init__(self, host="127.0.0.1", port=5555, *, timeout=120.0):
        # 延迟导入：只读取这个文件时，不需要加载 RLDX / PyTorch。
        import zmq
        from rldx.policy.server_client import PolicyClient

        self._client = PolicyClient(host=host, port=port)
        timeout_ms = max(1, int(timeout * 1000))
        self._client.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self._client.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._client.socket.setsockopt(zmq.LINGER, 0)
        self._session_id = uuid4().hex
        self._instruction = None
        self._fresh = True

        config = self._client.get_modality_config()
        self._video_deltas = tuple(config["video"].delta_indices)
        if not self._video_deltas or max(self._video_deltas) > 0:
            raise ValueError("Expected non-empty video offsets at or before the current step")
        if list(config["state"].delta_indices) != [0]:
            raise ValueError("This RoboCasa adapter supports only the current robot state")
        self._history = deque(maxlen=1 - int(min(self._video_deltas)))

    def start_episode(self, instruction, latest_obs):
        """显式开始新 episode / 子任务，即使指令文字与上一局相同。"""
        self._history.clear()
        self._instruction = instruction
        self._fresh = True
        self.observe(latest_obs)
        # 初始没有历史，用第一帧补齐，保持与评测 wrapper 相同的时间索引。
        self._history.extend([self._history[-1]] * (self._history.maxlen - 1))

    def observe(self, latest_obs):
        """每次仿真 step 后调用，保存这一时刻的观测快照。"""
        if self._instruction is None:
            raise RuntimeError("Call start_episode before observe")
        frame = {}
        for key, size in STATE_SIZES.items():
            state = np.array(latest_obs[key], dtype=np.float32, copy=True)
            if state.shape != (size,) or not np.isfinite(state).all():
                raise ValueError(f"Invalid state {key}: expected {size} finite values")
            frame[key] = state
        for key in VIDEO_KEYS:
            image = np.array(latest_obs[key], copy=True)
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"Invalid image {key}: expected uint8 RGB (H, W, 3)")
            frame[key] = image
        self._history.append(frame)

    def infer(self):
        """返回 {actions: (T, 12)}，不推进仿真，也不做底层控制转换。"""
        if not self._history:
            raise RuntimeError("Call start_episode before infer")
        frames = list(self._history)
        observation = {
            key: frames[-1][key][None, None] for key in STATE_SIZES
        }
        for key in VIDEO_KEYS:
            observation[key] = np.stack(
                [frames[int(delta) - 1][key] for delta in self._video_deltas]
            )[None]
        observation["annotation.human.task_description"] = [self._instruction]
        options = {"session_ids": [self._session_id], "reset_memory": [self._fresh]}
        actions, _info = self._client.get_action(observation, options=options)

        parts = []
        for key, size in ACTION_SIZES.items():
            value = np.asarray(actions[key], dtype=np.float32)
            if value.ndim != 3 or value.shape[0] != 1 or value.shape[2] != size:
                raise ValueError(f"Invalid action {key}: expected (1, T, {size}), got {value.shape}")
            parts.append(value[0])
        chunk = np.concatenate(parts, axis=-1)
        if not len(chunk) or not np.isfinite(chunk).all():
            raise ValueError("Expected a non-empty, finite action chunk")
        self._fresh = False
        return {"actions": chunk}

    def close(self):
        """清空本 session 的远端记忆，并释放本地连接。"""
        if self._client is None:
            return
        try:
            self._client.reset(options={"session_ids": [self._session_id]})
        finally:
            self._client.socket.close(linger=0)
            self._client.context.term()
            self._client = None

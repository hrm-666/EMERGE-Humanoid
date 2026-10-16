"""User-owned apple-table bring-up on SONIC's official loopback simulation interface.

The original MuJoCo owner and model are reused; the LightNav wrapper sets the
runtime start pose. No sudo or upstream source changes are involved.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .tabletop_reset import DEFAULT_START_DISTANCE_M

ROOT = Path(__file__).resolve().parents[3]


def require_loopback_simulation(args: argparse.Namespace) -> None:
    """Require local endpoints and the original scene/controller on interface lo."""
    endpoints = (args.observation_endpoint, args.sonic_feedback_endpoint,
                 args.sonic_bind)
    if any(not endpoint.startswith("tcp://127.0.0.1:")
           for endpoint in endpoints):
        raise RuntimeError(
            "loopback simulation requires all ZMQ endpoints on 127.0.0.1")
    if not args.lightnav_server.startswith("ws://127.0.0.1:"):
        raise RuntimeError(
            "loopback simulation requires the model on 127.0.0.1")
    current_namespace = os.stat("/proc/self/ns/net").st_ino
    scene_ready = False
    sonic_ready = False
    for process in Path("/proc").iterdir():
        if not process.name.isdecimal():
            continue
        try:
            if process.stat().st_uid != os.getuid():
                continue
            if (process / "ns/net").stat().st_ino != current_namespace:
                continue
            command = (process /
                       "cmdline").read_bytes().decode().rstrip("\0").split("\0")
        except (OSError, UnicodeError):
            continue
        scene_ready |= is_loopback_scene(command)
        sonic_ready |= is_loopback_sonic(command)
    if not scene_ready or not sonic_ready:
        raise RuntimeError(
            "loopback motion requires the original tabletop_pick.xml simulation and SONIC "
            "running as this user with interface lo; start local_tabletop sim and sonic first"
        )


def is_loopback_scene(command: list[str]) -> bool:
    required = {
        "--interface": "lo",
        "--domain": "0",
        "--observation-endpoint": "tcp://127.0.0.1:5560",
    }
    if not any(
            Path(item).name == "p5_sim_observation_server.py" or
            item == "integrations.lightnav_sonic_emerge.bridge.tabletop_sim"
            for item in command):
        return False
    for key, value in required.items():
        if key not in command or command[command.index(key) +
                                         1:command.index(key) + 2] != [value]:
            return False
    if "--model" not in command:
        return False
    model = command[command.index("--model") + 1:command.index("--model") + 2]
    return len(model) == 1 and Path(model[0]).name == "tabletop_pick.xml"


def is_loopback_sonic(command: list[str]) -> bool:
    return len(command) > 1 and Path(
        command[0]).name == "g1_deploy_onnx_ref" and command[1] == "lo"


def build_command(mode: str,
                  extra: list[str]) -> tuple[list[str], dict[str, str]]:
    """Build a simulation component command and its isolated environment.

    Args:
        mode: Component name: sim, sonic, control, server, ego, evaluate, or
            postprocess.
        extra: Additional CLI arguments, excluding fixed simulation settings.

    Returns:
        The argument vector and environment for the child process. Model code
        and weights resolve under LIGHTNAV_ROOT; artifacts stay in this repo.

    Raises:
        ValueError: A caller overrides fixed simulation settings or launches
            the graphical simulator without a display.
    """
    home = Path.home()
    bridge = Path(
        os.environ.get("VISIONPRO_SONIC_ROOT", home / "visionpro_sonic_bridge"))
    core = Path(os.environ.get("GR00T_ROOT", home / "GR00T-WholeBodyControl"))
    isaac = Path(os.environ.get("ISAAC_ROOT", home / "Isaac-GR00T-N1.7"))
    client_python = os.environ.get(
        "SONIC_PYTHON", str(home / "miniconda3/envs/sonic/bin/python"))
    model_python = os.environ.get(
        "LIGHTNAV_PYTHON", str(home / "miniconda3/envs/lightnav-0/bin/python"))
    env = os.environ.copy()
    lightnav_root = Path(env.get("LIGHTNAV_ROOT",
                                 home / "LightNav-0")).expanduser().resolve()
    env.update(PYTHONPATH=f"{ROOT}:{lightnav_root / 'src'}",
               PYTHONUNBUFFERED="1")
    if mode == "sim":
        if any(key in extra for key in ("--interface", "--domain", "--model",
                                        "--no-elastic-band")):
            raise ValueError(
                "local simulation fixes the original scene, support and lo/domain 0"
            )
        if "--headless" not in extra and not env.get("DISPLAY"):
            raise ValueError(
                "start sim on the laboratory desktop; DISPLAY is unset")
        env["PYTHONPATH"] += (
            f":{isaac}:{core}:{core / 'external_dependencies/unitree_sdk2_python'}"
            f":{bridge / 'integrations/inspire_mujoco'}:{bridge}")
        if "--headless" in extra:
            env["MUJOCO_GL"] = "egl"
        output = ROOT / "outputs/tabletop_scene"
        command = [
            str(core / ".venv_sim/bin/python"),
            "-m",
            "integrations.lightnav_sonic_emerge.bridge.tabletop_sim",
            "--model",
            str(bridge /
                "integrations/inspire_mujoco/scenes/tabletop_pick.xml"),
            "--interface",
            "lo",
            "--domain",
            "0",
            "--observation-endpoint",
            "tcp://127.0.0.1:5560",
            "--pose-endpoint",
            "tcp://127.0.0.1:5561",
            "--start-distance-m",
            str(DEFAULT_START_DISTANCE_M),
            "--observation-hz",
            "10",
            "--duration",
            "0",
            "--output-dir",
            str(output),
            "--screenshot",
            str(output / "final_third_person.png"),
            "--ego-screenshot",
            str(output / "final_ego.png"),
            "--metrics",
            str(output / "task_metrics.json"),
        ]
    elif mode in ("evaluate", "postprocess"):
        command = [
            str(core / ".venv_sim/bin/python"),
            "-m",
            ("integrations.lightnav_sonic_emerge.bridge.tabletop_evaluate"
             if mode == "evaluate" else
             "integrations.lightnav_sonic_emerge.bridge.tabletop_video"),
            "--model",
            str(bridge /
                "integrations/inspire_mujoco/scenes/tabletop_pick.xml"),
        ]
        env["MUJOCO_GL"] = "egl"
    elif mode == "sonic":
        if extra:
            raise ValueError(
                "the loopback SONIC launch has no additional arguments")
        deploy = core / "gear_sonic_deploy"
        output = ROOT / "outputs/tabletop_sonic"
        output.mkdir(parents=True, exist_ok=True)
        command = [
            "/usr/bin/bash",
            "-c",
            'source /opt/ros/humble/setup.bash; exec "$@"',
            "lightnav-loopback-sonic",
            str(deploy / "target/release/g1_deploy_onnx_ref"),
            "lo",
            str(deploy / "policy/release/model_decoder.onnx"),
            str(deploy / "reference/example"),
            "--obs-config",
            str(deploy / "policy/release/observation_config_zmq_ardy_g1.yaml"),
            "--encoder-file",
            str(deploy / "policy/release/model_encoder.onnx"),
            "--planner-file",
            str(deploy / "planner/target_vel/V2/planner_sonic.onnx"),
            "--input-type",
            "zmq_manager",
            "--output-type",
            "zmq",
            "--zmq-host",
            "127.0.0.1",
            "--zmq-port",
            "5556",
            "--zmq-out-port",
            "5557",
            "--disable-crc-check",
            "--disable-dex3-hands",
            "--logs-dir",
            str(output),
            "--enable-csv-logs",
        ]
    elif mode == "control":
        command = [
            client_python,
            "-m",
            "integrations.lightnav_sonic_emerge.bridge.tabletop",
            "--simulation-loopback",
            "--observation-endpoint",
            "tcp://127.0.0.1:5560",
        ]
    elif mode == "ego":
        command = [
            client_python,
            "-m",
            "integrations.lightnav_sonic_emerge.bridge.ego_viewer",
            "--observation-endpoint",
            "tcp://127.0.0.1:5560",
        ]
    else:
        env.update(
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            CUDA_VISIBLE_DEVICES=env.get("CUDA_VISIBLE_DEVICES", "0"),
            VLN_KV_CACHE_GIB=env.get("VLN_KV_CACHE_GIB", "2"),
            VLN_VLLM_ENFORCE_EAGER=env.get("VLN_VLLM_ENFORCE_EAGER", "1"),
        )
        command = [
            model_python,
            "-m",
            "lightnav.serving.ws_server",
            "--task",
            "vln",
            "--model_path",
            env.get("MODEL_PATH",
                    str(lightnav_root / "checkpoints/LightNav-0")),
            "--backend",
            "vllm_local",
            "--host",
            "127.0.0.1",
            "--port",
            "8051",
            "--max_batch_size",
            "1",
            "--gpu_memory_utilization",
            "0.65",
            "--record_dir",
            str(ROOT / "outputs/tabletop_model"),
        ]
    return command + extra, env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",
                        choices=("sim", "sonic", "control", "server", "ego",
                                 "evaluate", "postprocess"))
    args, extra = parser.parse_known_args()
    command, env = build_command(args.mode, extra)
    os.chdir(ROOT)
    print(
        f"SIMULATION_ONLY LOOPBACK mode={args.mode}; no sudo; DDS interface=lo",
        flush=True)
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()

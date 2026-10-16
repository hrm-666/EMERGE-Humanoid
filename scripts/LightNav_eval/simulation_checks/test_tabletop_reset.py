"""Verify runtime robot-only reset and non-blocking request semantics."""

import time
import unittest
from types import SimpleNamespace

import mujoco
import numpy as np
import zmq

from robot.vla.lightnav_sonic import lightnav_ws
from robot.vla.lightnav_sonic import official01_observation
from scripts.LightNav_eval import tabletop
from scripts.LightNav_eval import tabletop_reset
from scripts.LightNav_eval import tabletop_sim
from scripts.LightNav_eval import tabletop_start


class TabletopResetTests(unittest.TestCase):

    def setUp(self):
        model = mujoco.MjModel.from_xml_string('''
        <mujoco><worldbody>
          <body name="pelvis" pos="0 0 0.8"><freejoint/><geom size="0.1"/>
            <body pos="0 0 0.2"><joint name="arm"/><geom size="0.05"/></body>
          </body>
          <body name="tabletop_lab_table" pos="0.5 0 0.7">
            <geom type="box" size="0.368 0.587 0.026"/>
          </body>
          <body name="apple" pos="0.25 0 0.8"><freejoint/><geom size="0.03"/></body>
        </worldbody></mujoco>''')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        self.band = SimpleNamespace(enable=False, point=np.zeros(3))
        self.runner = SimpleNamespace(model=model,
                                      data=data,
                                      elastic_band=self.band,
                                      band_body_id=model.body("pelvis").id,
                                      viewer=None)
        tabletop_sim._set_start_pose(self.runner, 3.5)
        self.server = tabletop_sim.SimulationResetServer(
            self.runner, "tcp://127.0.0.1:*")
        endpoint = self.server.socket.getsockopt_string(zmq.LAST_ENDPOINT)
        self.client = tabletop_reset.TabletopResetClient(endpoint)

    def tearDown(self):
        self.client.close()
        self.server.close()

    def wait_for(self, predicate):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            self.server.tick()
            if predicate():
                return
            time.sleep(0.005)
        self.fail("reset transport did not respond")

    def test_robot_returns_to_nearer_start_objects_and_clock_persist(self):
        for enabled in (False, True):
            with self.subTest(support=enabled):
                data = self.runner.data
                self.band.enable = enabled
                original = self.server.initial_qpos.copy()
                data.qpos[:8] += 0.1
                data.qvel[:] = 2
                data.time = 12.5
                apple_qa = int(self.runner.model.jnt_qposadr[-1])
                data.qpos[apple_qa] = 0.42
                apple_pose = data.qpos[apple_qa:].copy()
                self.client.request()
                self.wait_for(lambda: self.server.phase == "stopping")
                self.assertEqual(data.qpos[0], original[0] + 0.1)
                self.server.deadline = 0
                self.server.tick()
                np.testing.assert_allclose(data.qpos[self.server.qpos_indices],
                                           original)
                np.testing.assert_allclose(data.qvel[self.server.qvel_indices],
                                           0)
                np.testing.assert_array_equal(data.qpos[apple_qa:], apple_pose)
                self.assertEqual(data.qvel[-1], 2)
                self.assertEqual(data.time, 12.5)
                self.assertTrue(self.band.enable)
                self.assertAlmostEqual(data.qpos[0], 0.5 - 0.368 - 3.5)
                self.server.deadline = 0
                self.server.tick()
                self.assertEqual(self.band.enable, enabled)
                result = []
                self.wait_for(lambda: result.append(self.client.poll()) or
                              result[-1] is not None)
                self.assertTrue(result[-1]["ok"])
                self.assertFalse(self.client.pending)

    def test_random_reset_preserves_scene_and_can_return_to_fixed(self):
        self.client.close()
        self.server.close()
        sampler = tabletop_start.RandomStartSampler(
            tabletop_start.table_footprint(self.runner.model, self.runner.data),
            123)
        self.server = tabletop_sim.SimulationResetServer(
            self.runner, "tcp://127.0.0.1:*", sampler)
        self.client = tabletop_reset.TabletopResetClient(
            self.server.socket.getsockopt_string(zmq.LAST_ENDPOINT))
        data = self.runner.data
        initial = data.qpos.copy()
        data.time = 99
        apple_qa = int(self.runner.model.jnt_qposadr[-1])
        data.qpos[apple_qa] = 0.42
        apple = data.qpos[apple_qa:].copy()
        positions = []
        for options in ({
                "start_mode": "random",
                "radius_m": 3
        }, {}, {
                "start_mode": "fixed"
        }):
            self.client.request(**options)
            self.wait_for(lambda: self.server.phase == "stopping")
            self.server.deadline = 0
            self.server.tick()
            positions.append(data.qpos[:2].copy())
            self.assertEqual(data.qpos[2], initial[2])
            self.assertEqual(data.qpos[7], initial[7])
            np.testing.assert_array_equal(data.qpos[apple_qa:], apple)
            self.assertEqual(data.time, 99)
            self.server.deadline = 0
            self.server.tick()
            reply = []
            self.wait_for(lambda: reply.append(self.client.poll()) or reply[-1]
                          is not None)
            self.assertTrue(reply[-1]["ok"])
        self.assertFalse(np.allclose(positions[0], positions[1]))
        np.testing.assert_array_equal(positions[2], initial[:2])
        # Reject an unusable disk before entering the reset/teleport phases.
        before = data.qpos.copy()
        self.client.request(start_mode="random", radius_m=0.3)
        reply = []
        self.wait_for(
            lambda: reply.append(self.client.poll()) or reply[-1] is not None)
        self.assertFalse(reply[-1]["ok"])
        self.assertEqual(self.server.phase, "idle")
        np.testing.assert_array_equal(data.qpos, before)

    def test_expired_request_never_moves_robot(self):
        original = self.runner.data.qpos.copy()
        socket = self.client.context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(self.client.endpoint)
        try:
            socket.send_json({
                "action": "reset",
                "request_id": "old",
                "expires_at": 0
            })
            self.wait_for(lambda: socket.poll(0))
            self.assertFalse(socket.recv_json()["ok"])
            np.testing.assert_array_equal(self.runner.data.qpos, original)
        finally:
            socket.close()

    def test_timeout_is_nonblocking_and_allows_retry(self):
        self.client.request()
        self.assertIsNone(self.client.poll())
        self.client.deadline = 0
        self.assertFalse(self.client.poll()["ok"])
        self.assertFalse(self.client.pending)
        self.client.request()
        self.assertTrue(self.client.pending)

    def test_reset_cancellation_rejects_old_reply_and_new_trial_runs(self):
        state = tabletop.TabletopControl(tabletop.build_parser().parse_args([]))
        state.navigate()
        previous = state.generation
        state.stop("simulation reset requested")
        reply = lightnav_ws.LightNavReply(0, ((0.5, 0, 0),), False)
        frame = official01_observation.ObservationFrame(b"jpeg", time.time(), 1)
        self.assertIn("cancelled", state.accept(previous, reply, frame, 0))
        self.assertTrue(state.tick(0, True).is_idle)
        state.navigate()
        state.accept(state.generation, reply, frame, 0)
        self.assertFalse(state.tick(0, True).is_idle)


if __name__ == "__main__":
    unittest.main()

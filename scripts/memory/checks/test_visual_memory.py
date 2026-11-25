"""Visual-memory integration checks using synthetic cameras and offline providers."""

from __future__ import annotations

import asyncio
import base64
import copy
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from PIL import Image

from Emerge.agent.context import ContextBuilder
from Emerge.agent.loop import AgentLoop
from Emerge.agent.memory import MemoryConsolidator, MemoryStore
from Emerge.agent.tools.embodied import EmbodiedActionTool
from Emerge.agent.tools.update_plan import UpdatePlanTool
from Emerge.agent.tools.view_image import ViewImageTool
from Emerge.agent.visual_monitor import VisualMonitor
from Emerge.bus.queue import MessageBus
from Emerge.config.schema import VisualMonitorConfig
from Emerge.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from Emerge.providers.custom_provider import CustomProvider
from Emerge.runtime.workspace import reset_workspace_context
from Emerge.session.manager import SessionManager
from Emerge.utils.action_queue import update_action_document
from Emerge.utils.helpers import estimate_message_tokens, estimate_prompt_tokens


class OfflineProvider(LLMProvider):
    def __init__(self):
        super().__init__()
        self.requests = []

    def get_default_model(self):
        return "offline-test"

    async def chat(self, **kwargs):
        self.requests.append(kwargs)
        return LLMResponse(content="Done")


class VisualMemoryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name)
        (self.workspace / "EMBODIED.md").write_text("Synthetic controller")
        self.provider = OfflineProvider()
        self.monitor = VisualMonitor(
            workspace=self.workspace, provider=self.provider, model="offline-test",
            poll_interval_seconds=0.001, verification_timeout_seconds=0.5,
            capture_interval_seconds=0.005, confirmations=1,
        )
        self.addAsyncCleanup(self.monitor.aclose)

    async def plan(self, subgoal="Lift apple"):
        result = await UpdatePlanTool(self.workspace).execute(
            update_kind="new_mission", mission="Lift apple",
            main_line=[{"id": 1, "subgoal": subgoal, "done_criterion": "Apple is held above the table",
                        "status": "active", "retries": 0}],
            pointer=1, branch_stack=[], change_summary="Start synthetic task",
        )
        self.assertFalse(result.startswith("Error"), result)

    def action(self, status="running", action_id="offline-action"):
        update_action_document(self.workspace / "ACTION.md", lambda _: {"actions": [{
            "id": action_id, "action_type": "infer_gr00t", "parameters": {"duration_s": 60},
            "status": status, "result": "Controller finished" if status == "completed" else "",
        }]})

    def observation(self, revision=1, color="red", expires_at=None):
        directory = self.workspace / "artifacts/observations"
        directory.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 24), color).save(directory / "ego.png")
        # A non-reference camera must not enter the monitor's evidence.
        Image.new("RGB", (32, 24), "black").save(directory / "other.png")
        (directory / "observation.json").write_text(json.dumps({
            "revision": revision, "captured_at": f"frame-{revision}", "episode_id": "synthetic",
            "expires_at": expires_at if expires_at is not None else time.time() + 60,
            "reference_view": "ego", "views": [
                {"name": "ego", "image_path": "artifacts/observations/ego.png"},
                {"name": "other", "image_path": "artifacts/observations/other.png"},
            ],
        }))

    @staticmethod
    async def reviews(trace, frames):
        return [{"snapshot": frame.number, "view": "ego", "outcome": "uncertain",
                 "description": "Synthetic current view", "key_evidence": False} for frame in frames]

    async def test_capture_sampling_is_bounded_and_uses_only_reference_camera(self):
        self.action()
        for revision in range(1, 100):
            self.observation(revision, (revision, 0, 0))
            self.monitor._poll_once()
        trace = self.monitor._traces["offline-action"]
        self.assertLessEqual(len(trace.frames), 24)
        self.assertEqual(trace.next_number, 33)
        self.assertEqual(trace.first.revision, 1)
        self.assertEqual(trace.latest.revision, 97)
        self.assertEqual(set(trace.first.images), {"ego"})

    async def test_expired_frame_does_not_hide_terminal_action(self):
        self.action()
        self.observation()
        self.monitor._poll_once()
        self.action("completed")
        self.observation(4, "blue", expires_at=0)
        self.monitor._poll_once()
        trace = self.monitor._traces["offline-action"]
        self.assertTrue(trace.ended)
        self.assertEqual(trace.latest.revision, 1)
        self.assertIn("unavailable", trace.note)

    async def test_capture_continues_during_slow_inference_and_archives_independent_images(self):
        await self.plan()
        self.action()
        self.observation()
        entered, release = asyncio.Event(), asyncio.Event()

        async def slow_review(trace, frames):
            entered.set()
            await release.wait()
            return await self.reviews(trace, frames)

        self.monitor._verify = slow_review
        self.monitor.start(lambda _: None, asyncio.get_running_loop())
        await asyncio.wait_for(entered.wait(), 1)
        self.observation(4, "blue")
        await asyncio.to_thread(self.monitor._poll_once)
        trace = self.monitor._traces["offline-action"]
        self.assertEqual(trace.latest.revision, 4)
        self.action("completed")
        release.set()
        entry = await asyncio.wait_for(self.monitor.finish_action("offline-action"), 2)
        self.assertIn("Execution status: completed", entry)
        self.assertIn("revision 1", entry)
        self.assertIn("revision 4", entry)
        (self.workspace / "artifacts/observations/ego.png").unlink()
        images = list((self.workspace / "memory/images").rglob("*.jpg"))
        self.assertGreaterEqual(len(images), 2)
        self.assertLessEqual(len(images), 4)
        for image in images:
            with Image.open(image) as captured:
                captured.verify()
        journal = (self.workspace / "memory/ACTION_JOURNAL.md").read_text()
        self.assertEqual(journal.count("## Action offline-action:"), 1)
        self.assertNotIn("offline-action", MemoryStore(self.workspace).get_memory_context())
        self.assertIn("按需查询", MemoryStore(self.workspace).get_memory_context())

    async def test_stale_plan_result_cannot_interrupt_current_action(self):
        await self.plan()
        self.action()
        self.observation()
        self.monitor._poll_once()
        signals = []
        self.monitor._publish = signals.append

        async def stale_review(trace, frames):
            plan_path = self.workspace / "PLAN.md"
            plan_path.write_text(plan_path.read_text().replace("Lift apple", "Inspect apple"))
            return [{"snapshot": frame.number, "view": "ego", "outcome": "achieved",
                     "description": "Previous stage was achieved", "key_evidence": True} for frame in frames]

        self.monitor._verify = stale_review
        work = asyncio.create_task(self.monitor._verification_loop())
        try:
            for _ in range(100):
                if self.monitor._traces["offline-action"].milestone:
                    break
                await asyncio.sleep(0.005)
            self.assertIsNotNone(self.monitor._traces["offline-action"].milestone)
            self.assertEqual(signals, [])
        finally:
            work.cancel()
            await asyncio.gather(work, return_exceptions=True)

    async def test_shutdown_preserves_unassessed_evidence(self):
        self.action()
        self.observation()
        self.monitor._verify = AsyncMock(side_effect=RuntimeError("offline model"))
        self.monitor.start(lambda _: None, asyncio.get_running_loop())
        await asyncio.to_thread(self.monitor._poll_once)
        await self.monitor.aclose()
        journal = (self.workspace / "memory/ACTION_JOURNAL.md").read_text()
        self.assertIn("uncertain", journal)
        self.assertIn("Monitor stopped before a terminal", journal)
        self.assertTrue(list((self.workspace / "memory/images").rglob("*.jpg")))

    async def test_view_image_round_trip_is_multimodal_and_wire_order_is_valid(self):
        self.observation()
        result = await ViewImageTool(workspace=self.workspace, allowed_dir=self.workspace).execute(
            "artifacts/observations/ego.png")
        context = ContextBuilder(self.workspace)
        messages = [{"role": "assistant", "content": None, "tool_calls": [
            ToolCallRequest("image-call", "view_image", {}).to_openai_tool_call(),
            ToolCallRequest("text-call", "read_file", {}).to_openai_tool_call(),
        ]}]
        context.add_tool_result(messages, "image-call", "view_image", result)
        context.add_tool_result(messages, "text-call", "read_file", "Current robot state")
        original = copy.deepcopy(messages)
        wire = LLMProvider._sanitize_empty_content(messages)
        self.assertEqual([m["role"] for m in wire], ["assistant", "tool", "tool", "user"])
        self.assertIsInstance(wire[1]["content"], str)
        self.assertEqual(messages, original)
        self.assertNotIn("base64", result.text)
        self.assertGreater(estimate_message_tokens(messages[1]), 4096)
        self.assertGreater(estimate_prompt_tokens(messages), 4096)
        data = base64.b64decode(result.content[0].image_url.split(",", 1)[1])
        self.assertTrue(data.startswith(b"\xff\xd8"))
        with self.assertRaises(PermissionError):
            await ViewImageTool(workspace=self.workspace, allowed_dir=self.workspace).execute("../outside.png")

    async def test_existing_custom_provider_receives_text_tool_result_and_user_image(self):
        provider = CustomProvider(api_key="offline-key", api_base="http://localhost:1/v1")
        self.addAsyncCleanup(provider.aclose)
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Seen", tool_calls=[], reasoning_content=None),
                                     finish_reason="stop")], usage=None,
        )
        create = AsyncMock(return_value=response)
        with patch.object(provider._client.chat.completions, "create", create):
            result = await provider.chat(messages=[{
                "role": "tool", "tool_call_id": "image-call", "name": "view_image", "content": [
                    {"type": "text", "text": "Observed ego camera"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9g="}},
                ],
            }])
        self.assertEqual(result.content, "Seen")
        wire = create.call_args.kwargs["messages"]
        self.assertEqual([m["role"] for m in wire], ["tool", "user"])
        self.assertEqual(wire[0]["content"], "Observed ego camera")

    async def test_new_preserves_journal_and_images(self):
        image = self.workspace / "memory/images/action/0000_ego.jpg"
        image.parent.mkdir(parents=True)
        image.write_bytes(b"archived evidence")
        journal = self.workspace / "memory/MEMORY.md"
        journal.write_text("Past action: [image](images/action/0000_ego.jpg)")
        reset_workspace_context(self.workspace)
        self.assertIn("Past action", journal.read_text())
        self.assertEqual(image.read_bytes(), b"archived evidence")
        self.assertEqual((self.workspace / "PLAN.md").read_text(), "")

    async def test_conversation_compaction_does_not_rewrite_visual_journal(self):
        store = MemoryStore(self.workspace)
        store.memory_file.write_text("Visual journal must survive compaction")
        sessions = SessionManager(self.workspace)
        session = sessions.get_or_create("cli:offline")
        for role, content in [("user", "Old task"), ("assistant", "Old result"), ("user", "Continue")]:
            session.add_message(role, content)
        self.provider.chat = AsyncMock(return_value=LLMResponse(content=None, tool_calls=[
            ToolCallRequest("summary-call", "save_context_summary", {"summary": "Old task was reviewed"})]))
        consolidator = MemoryConsolidator(
            self.workspace, self.provider, "offline-test", sessions, 100,
            ContextBuilder(self.workspace).build_messages, lambda: [],
        )
        with patch.object(consolidator, "estimate_session_prompt_tokens", side_effect=[(200, "mock"), (20, "mock")]):
            await consolidator.maybe_consolidate_by_tokens(session)
        self.assertEqual(store.memory_file.read_text(), "Visual journal must survive compaction")
        restored = SessionManager(self.workspace).get_or_create("cli:offline")
        self.assertEqual(restored.last_consolidated, 2)
        self.assertIn("Old task was reviewed", restored.get_history()[0]["content"])

    async def test_memory_failure_keeps_controller_result_and_original_parameters(self):
        tool = EmbodiedActionTool(self.workspace, visual_monitor=self.monitor)
        tool._wait_for_action_result = AsyncMock(return_value="Completed")
        self.monitor.finish_action = AsyncMock(side_effect=RuntimeError("archive unavailable"))
        parameters = {"opening_m": 0.05}
        result = await tool.execute("set_gripper", parameters, "Test memory integration")
        self.assertIn("Completed", result)
        self.assertIn("Visual memory unavailable", result)
        self.monitor.finish_action.assert_awaited_once()
        self.assertEqual(parameters, {"opening_m": 0.05})
        self.assertTrue(tool._wait_for_action_result.call_args.kwargs["allow_visual_completion"])

    async def test_record_only_mode_has_no_visual_judge_or_interrupt(self):
        self.monitor.record_only = True
        self.assertTrue(self.monitor.passive)
        self.monitor._verify = AsyncMock(side_effect=AssertionError("visual judge called"))
        self.observation()
        self.action()
        self.monitor._poll_once()
        self.monitor.start(lambda _: self.fail("record-only mode interrupted motion"), asyncio.get_running_loop())
        self.action(status="completed")
        result = await asyncio.wait_for(self.monitor.finish_action("offline-action"), 2)
        self.assertIn("Visual memory", result)
        self.assertTrue((self.workspace / "memory/actions/offline-action.json").is_file())
        self.monitor._verify.assert_not_awaited()

    async def test_agent_loop_with_legacy_response_returns_image_without_logging_pixels(self):
        self.observation()
        self.provider.chat = AsyncMock(side_effect=[
            LLMResponse(content="Inspect camera", tool_calls=[ToolCallRequest(
                "image-call", "view_image", {"path": "artifacts/observations/ego.png"})]),
            LLMResponse(content="Camera reviewed"),
        ])
        agent = AgentLoop(
            MessageBus(), self.provider, self.workspace,
            visual_monitor_config=VisualMonitorConfig(enabled=False),
        )
        events = []
        agent.on_event = lambda kind, data: events.append((kind, data))
        final, _, messages = await agent._run_agent_loop([{"role": "user", "content": "Inspect scene"}])
        self.assertEqual(final, "Camera reviewed")
        self.assertIsInstance(messages[2]["content"], list)
        self.assertNotIn("base64", json.dumps(events))
        self.assertIs(agent.tools.get("execute_robot_action").visual_monitor, agent.visual_monitor)


if __name__ == "__main__":
    unittest.main()

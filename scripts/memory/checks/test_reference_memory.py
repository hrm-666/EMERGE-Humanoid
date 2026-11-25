"""Offline checks for summary-first planning and citation-bound evidence reads."""

import asyncio
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from Emerge.agent.context import ContextBuilder
from Emerge.agent.loop import AgentLoop
from Emerge.agent.reference_memory import ReferenceMemory, sha256
from Emerge.agent.tools.filesystem import WriteFileTool
from Emerge.agent.tools.search_action_memory import SearchActionMemoryTool
from Emerge.agent.tools.search_reference_memory import SearchReferenceMemoryTool
from Emerge.bus.queue import MessageBus
from Emerge.config.schema import VisualMonitorConfig
from Emerge.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from scripts.memory.checks.reference_fixture import install


class ReferenceMemoryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.workspace = self.root / "workspace"
        self.source = self.root / "pilot"
        self.snapshot = self.source / "snapshots/test"
        self.snapshot.mkdir(parents=True)
        self.actions = self.workspace / "memory/actions"
        self.actions.mkdir(parents=True)
        images = self.workspace / "memory/images/a"
        images.mkdir(parents=True)
        image = images / "0000.jpg"
        Image.new("RGB", (8, 8), "red").save(image)
        action = self.actions / "a.json"
        action.write_text(json.dumps({"controller": "workspace", "action_id": "a", "run_id": "prior",
                                      "recorded_at": "2026-11-06", "started_at": "2026-11-06",
                                      "text": "POISON_RAW_OLD apple successful grasp",
                                      "images": [str(image.relative_to(self.workspace))]}))
        self.image_id = "image:prior:a:0000.jpg"
        self.action_id = "action:prior:a"
        evidence = {
            self.action_id: {"path": str(action), "run_id": "prior", "time": "2026-11-06"},
            self.image_id: {"path": str(image), "run_id": "prior", "time": "2026-11-06"},
        }
        self.files = [action, image]
        manifest = {"evidence": evidence,
                    "source_sha256": {str(path): sha256(path.read_bytes()) for path in self.files}}
        review = {"space": [{"kind": "object", "entities": ["apple"],
                             "statement": "历史发现位置：木桌桌面。", "confidence": "observed",
                             "evidence_ids": [self.image_id]}],
                  "experiences": [{"stage": "grasp_apple", "conditions": "桌面苹果已可见且已接近。",
                                   "effective": "prepare → grasp → hold_r2。", "ineffective": "未记录失败分支。",
                                   "recommendation": "携物移动前切hold_r2。", "confidence": "observed",
                                   "evidence_ids": [self.action_id]}]}
        (self.snapshot / "manifest.json").write_text(json.dumps(manifest))
        (self.snapshot / "memory.json").write_text(json.dumps(review))
        (self.source / "CURRENT.json").write_text('{"snapshot":"test"}')
        install(self.source, self.workspace, True)
        self.memory = ReferenceMemory(self.workspace)

    def audits(self):
        return [json.loads(line) for line in (self.workspace / "memory/access.jsonl").read_text().splitlines()]

    def test_context_loads_two_summaries_without_raw_history_or_pixels(self):
        with patch("Emerge.agent.action_memory.retrieve", side_effect=AssertionError("raw history loaded")):
            messages = ContextBuilder(self.workspace).build_messages([], "找苹果", run_id="new")
        self.assertIn("历史发现位置", messages[0]["content"])
        self.assertIn("hold_r2", messages[0]["content"])
        self.assertNotIn("POISON_RAW_OLD", messages[0]["content"])
        self.assertIsInstance(messages[-1]["content"], str)
        event = self.audits()[-1]
        self.assertEqual(event["event"], "reference_context")
        self.assertEqual(event["run_id"], "new")
        self.assertEqual(set(event["documents"]), {"SPACE.md", "EXPERIENCES.md"})
        self.assertEqual(event["images"], [])

    def test_summary_search_and_tag_expansion_have_separate_audits(self):
        summary = self.memory.search("苹果", run_id="new")
        self.assertIn("object:apple", summary.text)
        self.assertNotIn("POISON_RAW_OLD", summary.text)
        self.assertEqual(summary.content, ())
        tool = SearchReferenceMemoryTool(self.workspace)
        tool.run_id = "new"
        result = asyncio.run(tool.execute(reference_ids=["stage:grasp_apple"], include_evidence=True))
        self.assertIn("POISON_RAW_OLD", result.text)
        self.assertEqual(sum(type(block).__name__ == "ImageContent" for block in result.content), 1)
        self.assertEqual(self.audits()[-1]["event"], "reference_evidence")
        self.assertTrue(self.audits()[-1]["images"])
        denied = self.memory.search("apple", include_evidence=True)
        self.assertEqual(denied.content, ())
        self.assertIn("Choose", denied.text)
        self.assertIn("unknown", self.memory.search(reference_ids=["E999"], include_evidence=True).text)

    def test_legacy_historical_tool_redirects_but_current_run_is_available(self):
        tool = SearchActionMemoryTool(self.workspace)
        tool.run_id = "new"
        result = asyncio.run(tool.execute("apple", scope="previous_runs"))
        self.assertNotIn("POISON_RAW_OLD", result.text)
        self.assertEqual(result.content, ())
        current = self.actions / "new.json"
        current.write_text(json.dumps({"controller": "workspace", "run_id": "new", "action_id": "new",
                                       "recorded_at": "2026-11-07", "text": "CURRENT apple attempt", "images": []}))
        self.assertIn("CURRENT", asyncio.run(tool.execute("apple", scope="current_run")).text)

    def test_route_query_and_explicit_frame_precede_old_evidence_with_same_read_limits(self):
        root, bundle = self.memory.load()
        # A legacy snapshot can retain stale citations before newer route views.
        for i in range(10):
            label, image_label = f'E9{i:03d}', f'E8{i:03d}'
            raw = json.dumps({'text': f'ROUTE_RECORD_{i}'}).encode()
            action_file = f'route-{i}.json'
            (root / action_file).write_bytes(raw)
            image_file = f'route-{i}.jpg'
            Image.new('RGB', (8, 8), (i * 20, 50, 100)).save(root / image_file)
            raw_image = (root / image_file).read_bytes()
            bundle['evidence'][label] = {'kind': 'action', 'evidence_id': f'route-{i}',
                'file': action_file, 'sha256': sha256(raw), 'image_labels': [image_label]}
            bundle['evidence'][image_label] = {'kind': 'image', 'evidence_id': f'frame-{i}',
                'file': image_file, 'sha256': sha256(raw_image), 'action_label': label}
        bundle['rows'].append({'id': 'stage:route', 'confidence': 'observed',
            'effective': '门框入口→左侧走廊→到达路口E9006；隔断端部→右侧出口→柜前E9008。柜侧黑把手→沿柜侧靠近E9009',
            'evidence_ids': [f'E9{i:03d}' for i in range(10)]})
        (root / 'bundle.json').write_text(json.dumps(bundle, ensure_ascii=False))
        result = self.memory.search(reference_ids=['stage:route'], query='隔断端部出口',
                                    include_evidence=True)
        self.assertIn('ROUTE_RECORD_8', result.text)
        event = self.audits()[-1]
        self.assertEqual(event['evidence_ids'][0], 'E9008')
        self.assertEqual(event['images'][0], 'E8008')
        self.assertEqual(len(event['images']), 2)
        self.assertEqual(len(event['evidence_ids']) - len(event['images']), 4)
        self.memory.search(reference_ids=['stage:route'], query='柜侧黑把手', include_evidence=True)
        self.assertEqual(self.audits()[-1]['images'][0], 'E8009')
        self.memory.search(reference_ids=['stage:route', 'E8009'],
                           query='隔断端部出口', include_evidence=True)
        self.assertEqual(self.audits()[-1]['images'][0], 'E8009')

    def test_content_dedup_survives_new_records_labels_and_snapshot(self):
        tool = SearchReferenceMemoryTool(self.workspace)
        tool.run_id = 'new'

        async def exercise():
            first = await tool.execute(reference_ids=['stage:grasp_apple'], include_evidence=True)
            self.assertTrue(first.content)
            root, bundle = self.memory.load()
            image = next(item for item in bundle['evidence'].values() if item['kind'] == 'image')
            # A new action/label and snapshot reuse the same actual image bytes.
            raw = json.dumps({'text': 'NEW_RECORD_WITH_SAME_IMAGE'}).encode()
            (root / 'new-action.json').write_bytes(raw)
            bundle['evidence']['E9998'] = {'kind': 'image', **image, 'action_label': 'E9999'}
            bundle['evidence']['E9999'] = {'kind': 'action', 'file': 'new-action.json',
                'sha256': sha256(raw), 'evidence_id': 'new-action', 'image_labels': ['E9998']}
            new_root = root.parent / 'changed'
            shutil.copytree(root, new_root)
            bundle['snapshot'] = 'changed'
            (new_root / 'bundle.json').write_text(json.dumps(bundle))
            (self.workspace / 'memory/reference/CURRENT.json').write_text('{"snapshot":"changed"}')
            second = await tool.execute(reference_ids=['E9999'], include_evidence=True)
            self.assertFalse(second.content)
            self.assertIn('NEW_RECORD_WITH_SAME_IMAGE', second.text)
            self.assertEqual(self.audits()[-1]['reused_images'], ['E9998'])
            self.assertEqual(self.audits()[-1]['images'], [])
            third = await tool.execute(reference_ids=['E9998', 'E9999'], include_evidence=True)
            self.assertFalse(third.content)
            self.assertNotIn('NEW_RECORD_WITH_SAME_IMAGE', third.text)
            self.assertEqual(self.audits()[-1]['event'], 'reference_reused')
            fourth = await tool.execute(reference_ids=['E9998'], include_evidence=True, reread_images=True)
            self.assertTrue(fourth.content)
            self.assertEqual(self.audits()[-1]['images'], ['E9998'])
            tool.run_id = 'another-task'
            self.assertTrue((await tool.execute(reference_ids=['E9998'], include_evidence=True)).content)

        asyncio.run(exercise())

    def test_route_comparison_survives_local_motion_and_returns_pending_step(self):
        from Emerge.agent.tools.update_plan import UpdatePlanTool
        plan = UpdatePlanTool(self.workspace)
        state = {'mission': 'Find cabinet', 'main_line': [{'id': 1, 'subgoal': 'Find cabinet',
                 'done_criterion': 'Visible', 'status': 'active', 'retries': 0}],
                 'pointer': 1, 'branch_stack': []}
        tool = SearchReferenceMemoryTool(self.workspace)
        tool.run_id = 'new'
        label = next(key for key, item in self.memory.load()[1]['evidence'].items() if item['kind'] == 'image')
        comparison = {'evidence_ids': [label], 'match': 'divider right, passage left',
                      'exit': 'left passage', 'arrival_cue': 'open floor at divider end'}

        async def exercise():
            await tool.execute(reference_ids=['stage:grasp_apple'], include_evidence=True)
            plan._atomic_write(plan._render(state, 'now', 'new_mission', 'start'))
            plan.advance_route('entry', 'Enter left passage', segment_event='exit_selected', comparison=comparison)
            self.assertFalse((await tool.execute(reference_ids=['stage:grasp_apple'], include_evidence=True)).content)
            segment = tool._segment
            for i in range(3):
                plan.advance_route(f'Camera {i}: slight turn', f'Sidestep then continue left {i}')
                result = await tool.execute(reference_ids=['stage:grasp_apple'], include_evidence=True)
                self.assertFalse(result.content)
                self.assertIn('left passage', result.text)
                self.assertIn(f'Sidestep then continue left {i}', result.text)
                self.assertEqual(segment, tool._segment)
            # The compact decision remains visible after image tool messages leave the window.
            from Emerge.agent.planning_context import PlanningContext
            from Emerge.config.schema import PlanningMemoryConfig
            context = ContextBuilder(self.workspace)
            projected = PlanningContext(context, PlanningMemoryConfig(), lambda *args: None).project(
                [{'role': 'system', 'content': ''}, {'role': 'user', 'content': 'Continue'}], 1)
            self.assertIn('open floor at divider end', projected[0]['content'])
            plan.advance_route('blocked passage', 'Choose another visible exit', segment_event='route_invalidated')
            result = await tool.execute(reference_ids=['stage:grasp_apple'], include_evidence=True)
            self.assertNotEqual(segment, tool._segment)
            self.assertFalse(result.content)
            self.assertNotIn('Retained comparison', result.text)
            self.assertIn('Previous route invalidated', result.text)
            self.assertIn('Choose another visible exit', result.text)

        asyncio.run(exercise())

    def test_decision_log_distinguishes_loading_from_model_declared_use(self):
        self.assertIsNone(self.memory.record_decision("update_plan", {"change_summary": "探索当前场景"}, "new"))
        result = self.memory.record_decision("execute_robot_action", {
            "reasoning": "参考 stage:grasp_apple，移动前切hold_r2。"}, "new")
        self.assertEqual(result["basis"], "model_declared")
        self.assertEqual(result["reference_ids"], ["stage:grasp_apple"])
        self.assertEqual(self.audits()[-1]["event"], "reference_decision")

    def test_package_survives_source_removal_and_workspace_memory_reset(self):
        for path in self.files:
            path.unlink()
        shutil.rmtree(self.workspace / "memory")
        install(self.source, self.workspace, True)
        result = self.memory.search(reference_ids=["object:apple"], include_evidence=True)
        self.assertIn("POISON_RAW_OLD", result.text)
        self.assertTrue(result.content)

    def test_changed_evidence_and_foreign_controller_do_not_trigger_raw_fallback(self):
        root, bundle = self.memory.load()
        label = next(key for key, item in bundle["evidence"].items() if item["kind"] == "image")
        (root / bundle["evidence"][label]["file"]).write_bytes(b"modified")
        result = self.memory.search(reference_ids=[label], include_evidence=True)
        self.assertIn("changed", result.text)
        self.assertEqual(result.content, ())
        (self.workspace / "ROBOT_STATE.md").write_text(json.dumps({"robots": {"different": {"runtime": "real"}}}))
        context = ContextBuilder(self.workspace).build_messages([], "apple", run_id="new")
        self.assertIn("mismatch", context[0]["content"])
        self.assertIsInstance(context[-1]["content"], str)
        self.assertNotIn("POISON_RAW_OLD", context[0]["content"])

    def test_reference_write_protection(self):
        tool = WriteFileTool(workspace=self.workspace, protected_paths={self.workspace / "memory/reference"})
        result = asyncio.run(tool.execute("memory/reference/SPACE.md", "replacement"))
        self.assertIn("read-only", result)

    def test_agent_loop_exposes_pixels_and_emits_declared_reference_use(self):
        requests, events = [], []

        class OfflineProvider(LLMProvider):
            def get_default_model(self):
                return "offline"

            async def chat(self, **kwargs):
                requests.append(kwargs)
                if len(requests) == 1:
                    return LLMResponse(None, [ToolCallRequest("ref", "search_reference_memory", {
                        "reference_ids": ["stage:grasp_apple"], "include_evidence": True,
                    })])
                if len(requests) == 2:
                    return LLMResponse(None, [ToolCallRequest("plan", "update_plan", {
                        "update_kind": "new_mission", "mission": "Offline memory check", "pointer": 1,
                        "main_line": [{"id": 1, "subgoal": "Inspect apple", "done_criterion": "Apple visible",
                                       "status": "active", "retries": 0}], "branch_stack": [],
                        "change_summary": "参考 stage:grasp_apple，携物移动前切hold_r2。",
                    })])
                return LLMResponse("Offline check done")

        loop = AgentLoop(bus=MessageBus(), provider=OfflineProvider(), workspace=self.workspace,
                         visual_monitor_config=VisualMonitorConfig(enabled=False))
        loop.run_id = "offline"
        loop.on_event = lambda kind, data: events.append((kind, data))
        loop._start_visual_monitor()
        self.assertIn("search_reference_memory", [tool["function"]["name"] for tool in loop.tools.get_definitions()])
        asyncio.run(loop._run_agent_loop(ContextBuilder(self.workspace).build_messages([], "Inspect", run_id="offline")))
        # The archived image is in a multimodal tool result for the next model call.
        tool_reply = next(m for m in requests[1]["messages"] if m.get("tool_call_id") == "ref")
        self.assertTrue(any(block["type"] == "image_url" for block in tool_reply["content"]))
        self.assertTrue(any(kind == "memory.reference_decision" and
                            data["reference_ids"] == ["stage:grasp_apple"] for kind, data in events))
        self.assertTrue(any(e["event"] == "reference_evidence" and e["run_id"] == "offline" for e in self.audits()))


if __name__ == "__main__":
    unittest.main()

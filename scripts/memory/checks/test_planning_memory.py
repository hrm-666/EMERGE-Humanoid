"""Offline regression checks for incremental memory and bounded multimodal requests."""
import copy
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from Emerge.agent.context import ContextBuilder
from Emerge.agent.loop import AgentLoop
from Emerge.agent.planning_context import PlanningContext
from Emerge.agent.planning_memory import PlanningMemory, merge_stage_evidence
from Emerge.agent.reference_memory import ReferenceMemory
from Emerge.base import Tool
from Emerge.bus.queue import MessageBus
from Emerge.config.schema import PlanningMemoryConfig, VisualMonitorConfig
from Emerge.providers.base import LLMProvider, LLMResponse, ToolCallRequest


class MemoryProvider(LLMProvider):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.fail = False
        self.row_text = '苹果最后确认在桌面；先找到桌子，再接近。'

    def get_default_model(self):
        return 'offline'

    async def chat(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        if self.fail:
            raise ValueError('offline update failure')
        payload = json.loads(kwargs['messages'][1]['content'][0]['text'])
        label = payload['new_evidence'][-1]['label']
        return LLMResponse(None, [ToolCallRequest('merge', 'update_planning_memory', {
            'rows': [{'id': 'object:apple', 'text': self.row_text, 'confidence': 'observed',
                      'evidence_ids': [label]}], 'remove_ids': [], 'working_summary': '已找到苹果，下一步对位。'
        })], usage={'prompt_tokens': 123, 'completion_tokens': 45})


class PlanningMemoryTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.workspace = Path(self.tmp.name)
        self.provider = MemoryProvider()
        self.events = []
        self.config = PlanningMemoryConfig(minimum_interval_seconds=0)
        self.memory = PlanningMemory(self.workspace, self.provider, 'offline', self.config,
                                     lambda kind, data: self.events.append((kind, data)))
        self.messages = [{'role': 'system', 'content': 'rules'}, {'role': 'user', 'content': '抓取苹果'}]
        self.memory.refresh_working(self.messages, 1, 'run')

    def record(self, number, text='苹果位于桌面。', controller='workspace'):
        root = self.workspace / 'memory'
        (root / 'actions').mkdir(exist_ok=True)
        image = root / f'images/a{number}/last.jpg'
        image.parent.mkdir(parents=True)
        Image.new('RGB', (8, 8), 'red').save(image)
        (root / f'actions/a{number}.json').write_text(json.dumps({
            'action_id': f'a{number}', 'controller': controller, 'run_id': 'run',
            'recorded_at': f'2026-11-07T10:00:{number:02d}', 'text': text,
            'images': [str(image.relative_to(self.workspace))]}, ensure_ascii=False))

    def test_legacy_memory_is_archived_before_working_summary_replacement(self):
        target = self.workspace / 'memory/MEMORY.md'
        target.write_text('Previously learned user and environment facts')
        self.memory.refresh_working(self.messages, 1, 'run')
        archives = list((self.workspace / 'memory/archive').glob('MEMORY-before-planning-*.md'))
        self.assertEqual(len(archives), 1)
        self.assertEqual(archives[0].read_text(), 'Previously learned user and environment facts')
        self.assertTrue(target.read_text().startswith('# MEMORY — 当前任务'))

    async def test_batches_and_merges_one_canonical_object_with_citable_evidence(self):
        for i in range(11):
            self.record(i)
        await self.memory.maybe_update(run_id='run')
        self.assertEqual(len(self.provider.calls), 0)
        self.record(11)
        await self.memory.maybe_update(run_id='run')
        self.assertEqual(len(self.provider.calls), 1)
        self.assertEqual(self.memory.pending(), [])
        reference = ReferenceMemory(self.workspace)
        first_root, first = reference.load()
        self.assertEqual(len(first['rows']), 1)
        old_label = first['rows'][0]['evidence_ids'][0]
        expanded = reference.search(reference_ids=['object:apple'], include_evidence=True)
        self.assertIn('苹果位于桌面', expanded.text)
        self.assertEqual(len([b for b in expanded.content if type(b).__name__ == 'ImageContent']), 1)
        for i in range(12, 17):
            self.record(i, '苹果已经放入抽屉。')
        self.provider.row_text = '苹果最后确认在柜子抽屉内；桌面是此前位置。'
        await self.memory.maybe_update(run_id='run', force=True)
        root, bundle = reference.load()
        self.assertNotEqual(root, first_root)
        self.assertEqual(len(bundle['rows']), 1)
        self.assertIn('抽屉', (root / 'SPACE.md').read_text())
        self.assertEqual((root / 'SPACE.md').read_text().count('[object:apple]'), 1)
        self.assertTrue(first_root.exists())
        self.assertIn(old_label, first['evidence'])
        self.assertTrue(any(k == 'memory.update_usage' and d['usage']['prompt_tokens'] == 123 for k, d in self.events))
        loaded = ContextBuilder(self.workspace).build_messages([], '下一步')
        self.assertIsInstance(loaded[-1]['content'], str)
        self.assertIn('抽屉', loaded[0]['content'])

    async def test_stage_checkpoint_requires_five_actions_and_respects_cooldown(self):
        self.config.minimum_interval_seconds = 60
        self.memory.stage_changed = True
        self.memory.last_attempt = time.monotonic() - 10000
        for i in range(4):
            self.record(i)
            await self.memory.maybe_update(run_id='run')
            self.assertEqual(len(self.provider.calls), 0)
        self.record(4)
        await self.memory.maybe_update(run_id='run')
        self.assertEqual(len(self.provider.calls), 1)
        for i in range(5, 10):
            self.record(i)
        self.memory.stage_changed = True
        await self.memory.maybe_update(run_id='run')
        self.assertEqual(len(self.provider.calls), 1)
        self.memory.last_attempt -= 61
        await self.memory.maybe_update(run_id='run')
        self.assertEqual(len(self.provider.calls), 2)

    async def test_elapsed_time_does_not_trigger_regular_update(self):
        # Legacy interval settings are ignored by the schema and scheduler.
        self.config = PlanningMemoryConfig(update_interval_seconds=1,
                                           minimum_interval_seconds=0)
        self.memory.config = self.config
        for i in range(11):
            self.record(i)
        with patch('Emerge.agent.planning_memory.time.monotonic', return_value=10**12):
            await self.memory.maybe_update()
            self.assertEqual(self.provider.calls, [])
            self.record(11)
            await self.memory.maybe_update()
        self.assertEqual(len(self.provider.calls), 1)

    async def test_forced_flush_keeps_small_tail_pending_across_restart(self):
        for i in range(4):
            self.record(i)
            await self.memory.maybe_update(force=True)
        self.assertEqual(self.provider.calls, [])
        self.assertEqual(len(self.memory.pending()), 4)
        renewed = PlanningMemory(self.workspace, self.provider, 'offline', self.config,
                                lambda kind, data: self.events.append((kind, data)))
        renewed.refresh_working(self.messages, 1, 'next-run')
        self.record(4)
        await renewed.maybe_update(force=True)
        self.assertEqual(len(self.provider.calls), 1)
        self.assertEqual(renewed.pending(), [])

    async def test_final_flush_commits_one_to_four_actions_without_duplicate_calls(self):
        for count in range(1, 5):
            with self.subTest(count=count):
                for i in range(count):
                    self.record(count * 10 + i)
                previous = len(self.provider.calls)
                await self.memory.maybe_update(force=True)
                self.assertEqual(len(self.provider.calls), previous)
                await self.memory.flush(run_id='run')
                self.assertEqual(len(self.provider.calls), previous + 1)
                self.assertEqual(self.memory.pending(), [])
                await self.memory.flush(run_id='run')
                self.assertEqual(len(self.provider.calls), previous + 1)

    async def test_final_flush_drains_large_backlog_and_short_last_batch(self):
        for i in range(35):
            self.record(i)
        await self.memory.flush(run_id='run')
        sizes = [len(json.loads(call['messages'][1]['content'][0]['text'])['new_evidence'])
                 for call in self.provider.calls]
        self.assertEqual(sizes, [16, 16, 3])
        self.assertEqual(self.memory.pending(), [])

    async def test_failed_final_flush_retains_tail_and_cools_down_shutdown_retry(self):
        self.config.minimum_interval_seconds = 60
        self.record(1)
        self.provider.fail = True
        await self.memory.flush(run_id='run')
        self.assertEqual(len(self.provider.calls), 1)
        self.assertEqual(len(self.memory.pending()), 1)
        await self.memory.flush(run_id='run')
        self.assertEqual(len(self.provider.calls), 1)
        self.provider.fail = False
        self.memory.last_attempt -= 61
        await self.memory.flush(run_id='run')
        self.assertEqual(self.memory.pending(), [])

    async def test_final_flush_retries_a_regular_failure_once_within_cooldown(self):
        self.config.minimum_interval_seconds = 60
        for i in range(5):
            self.record(i)
        self.provider.fail = True
        await self.memory.maybe_update(force=True)
        self.assertEqual(len(self.provider.calls), 1)
        self.provider.fail = False
        await self.memory.flush(run_id='run')
        self.assertEqual(len(self.provider.calls), 2)
        self.assertEqual(self.memory.pending(), [])

    async def test_completed_task_flushes_tail_before_returning(self):
        class MainProvider(MemoryProvider):
            async def chat(self, **kwargs):
                return LLMResponse('Task complete.')

        loop = AgentLoop(MessageBus(), MainProvider(), self.workspace,
                         visual_monitor_config=VisualMonitorConfig(enabled=False))
        loop.planning_memory = self.memory
        self.record(1, '抽屉开度6.9毫米，双臂归位，任务完成。')
        self.provider.row_text = '苹果在抽屉内，抽屉已关闭，双臂归位。'
        output = await loop.process_direct('完成放置任务')
        self.assertEqual(output, 'Task complete.')
        self.assertEqual(self.memory.pending(), [])
        self.assertIn('抽屉已关闭', (self.memory.reference.directory / 'SPACE.md').read_text())
        await loop.aclose()
        self.assertEqual(len(self.provider.calls), 1)

    async def test_status_patch_keeps_success_route_and_its_evidence(self):
        for i in range(5):
            self.record(i)
        route = '桌边后退→沿隔断到端部→绕行到柜子。'

        async def reply(**kwargs):
            payload = json.loads(kwargs['messages'][1]['content'][0]['text'])
            label = payload['new_evidence'][0]['label']
            row = {'id': 'stage:table_to_cabinet', 'text': '正在转运',
                   'confidence': 'observed', 'evidence_ids': [label]}
            if not self.memory.reference.installed:
                row.update(conditions='桌边持果', effective=route,
                           ineffective='贴隔断侧行时卡住', recommendation='沿端部空地绕行')
            else:
                row.update(text='本轮尚未证实退出墙角', ineffective='桌脚旁转向受阻')
            return LLMResponse(None, [ToolCallRequest('merge', 'update_planning_memory', {
                'rows': [row], 'remove_ids': [], 'working_summary': '继续绕行'})])

        with patch.object(self.provider, 'chat_with_retry', side_effect=reply):
            await self.memory.maybe_update(force=True)
            _, first = self.memory.reference.load()
            old_label = first['rows'][0]['evidence_ids'][0]
            for i in range(5, 10):
                self.record(i)
            await self.memory.maybe_update(force=True)
        root, bundle = self.memory.reference.load()
        doc = (root / 'EXPERIENCES.md').read_text()
        self.assertIn(route, doc)
        self.assertIn('桌脚旁转向受阻', doc)
        self.assertNotIn('尚未证实', doc)
        self.assertIn(old_label, bundle['rows'][0]['evidence_ids'])
        self.assertIn(old_label, bundle['evidence'])
        expanded = self.memory.reference.search(reference_ids=['stage:table_to_cabinet'], include_evidence=True)
        self.assertNotIn('Error:', expanded.text)

    async def test_api_failure_preserves_pointer_and_pending_then_recovers(self):
        for i in range(5):
            self.record(i)
        await self.memory.maybe_update(force=True)
        pointer = self.memory.reference.directory / 'CURRENT.json'
        before = pointer.read_bytes()
        for i in range(5, 10):
            self.record(i)
        self.provider.fail = True
        await self.memory.maybe_update(force=True)
        self.assertEqual(pointer.read_bytes(), before)
        self.assertEqual(len(self.memory.pending()), 5)
        self.assertTrue(any(k == 'memory.update_warning' for k, _ in self.events))
        self.provider.fail = False
        await self.memory.maybe_update(force=True)
        self.assertEqual(self.memory.pending(), [])

    async def test_unknown_evidence_rejected_without_overwriting_memory(self):
        for i in range(5):
            self.record(i)
        await self.memory.maybe_update(force=True)
        before = (self.memory.reference.directory / 'CURRENT.json').read_bytes()
        for i in range(5, 10):
            self.record(i)
        response = LLMResponse(None, [ToolCallRequest('bad', 'update_planning_memory', {
            'rows': [{'id': 'object:apple', 'text': 'invented', 'confidence': 'observed',
                      'evidence_ids': ['E99999']}], 'remove_ids': [], 'working_summary': ''})])
        with patch.object(self.provider, 'chat_with_retry', return_value=response):
            await self.memory.maybe_update(force=True)
        self.assertEqual((self.memory.reference.directory / 'CURRENT.json').read_bytes(), before)
        self.assertEqual(len(self.memory.pending()), 5)

    async def test_two_images_max_foreign_controller_excluded_and_no_legacy_api(self):
        for i in range(12):
            self.record(i)
        self.record(12, controller='other:real')
        await self.memory.maybe_update()
        content = self.provider.calls[0]['messages'][1]['content']
        self.assertEqual(sum(b['type'] == 'image_url' for b in content), 2)
        self.assertEqual(len(json.loads(content[0]['text'])['new_evidence']), 12)
        old = self.workspace / 'memory/MEMORY.md'
        old.write_text('# Visual Action Memory\n<!-- visual-action:old --> OLD_JOURNAL')
        self.memory.refresh_working(self.messages, 1, 'run')
        self.assertIn('OLD_JOURNAL', next((self.workspace / 'memory/archive').glob('*.md')).read_text())
        self.assertNotIn('OLD_JOURNAL', old.read_text())

    async def test_route_entry_arrival_frames_replace_batch_endpoints_without_extra_call(self):
        for i in range(12):
            self.record(i)
        for number, purpose, result in ((4, 'exploration', 'exploration_progress'),
                                        (7, 'exploration', 'motion_stalled'),
                                        (9, 'exploration', 'visual_turn_finished'),
                                        (10, 'scan', 'scan_segment_finished')):
            path = self.workspace / f'memory/actions/a{number}.json'
            record = json.loads(path.read_text())
            entry = self.workspace / f'memory/images/a{number}/entry.jpg'
            Image.new('RGB', (8, 8), 'blue').save(entry)
            record['images'].insert(0, str(entry.relative_to(self.workspace)))
            record['text'] = (f'## Action a{number}: navigate\n'
                              f'- Parameters: `{{"purpose": "{purpose}"}}`\n'
                              '- Action intent: 沿可见通道到出口并观察下一路标\n'
                              f'- Controller result: Completed: {{"reason": "{result}"}}')
            path.write_text(json.dumps(record))
        await self.memory.maybe_update()
        self.assertEqual(len(self.provider.calls), 1)
        content = self.provider.calls[0]['messages'][1]['content']
        payload = json.loads(content[0]['text'])
        self.assertEqual(len(payload['new_evidence']), 12)
        self.assertEqual(sum(block['type'] == 'image_url' for block in content), 2)
        event = next(data for kind, data in self.events if kind == 'memory.update_started')
        row = payload['new_evidence'][4]
        self.assertEqual(event['image_labels'], [row['entry_image_label'], row['image_label']])
        # The first frame is genuinely before travel, not a second copy of the last view.
        self.assertNotEqual(content[2]['image_url']['url'], content[4]['image_url']['url'])

    def test_stage_evidence_replaces_stale_prefix_and_keeps_success_support_bounded(self):
        evidence = {f'E{i:05d}': {} for i in range(1, 20)}
        old = {'effective': '入口E00005→出口E00006',
               'evidence_ids': [f'E{i:05d}' for i in range(1, 7)]}
        row = {'effective': old['effective'], 'evidence_ids': ['E00007']}
        merge_stage_evidence(row, old, evidence)
        self.assertEqual(row['evidence_ids'][:3], ['E00005', 'E00006', 'E00007'])
        self.assertEqual(len(row['evidence_ids']), 6)
        improved = {'effective': '新入口E00008→出口E00009', 'evidence_ids': ['E00009', 'E00008']}
        merge_stage_evidence(improved, old, evidence)
        self.assertEqual(improved['evidence_ids'][:2], ['E00008', 'E00009'])
        self.assertLessEqual(len(improved['evidence_ids']), 6)

    async def test_two_stage_updates_commit_route_support_and_new_findings_with_six_refs(self):
        async def reply(**kwargs):
            payload = json.loads(kwargs['messages'][1]['content'][0]['text'])
            labels = [r['label'] for r in payload['new_evidence']]
            first = not payload['existing_rows']
            route = (f'入口{labels[-2]}→出口{labels[-1]}' if first
                     else payload['existing_rows'][0]['effective'])
            return LLMResponse(None, [ToolCallRequest('merge', 'update_planning_memory', {
                'rows': [{'id': 'stage:route', 'confidence': 'observed',
                          'effective': route, 'ineffective': '本次绕行受阻',
                          'evidence_ids': labels[:6] if first else labels[:1]}],
                'remove_ids': [], 'working_summary': '沿已选出口继续'})])

        with patch.object(self.provider, 'chat_with_retry', side_effect=reply):
            for i in range(6):
                self.record(i)
            await self.memory.maybe_update(force=True)
            _, first = self.memory.reference.load()
            support = first['rows'][0]['evidence_ids'][:2]
            for i in range(6, 11):
                self.record(i)
            await self.memory.maybe_update(force=True)
        _, latest = self.memory.reference.load()
        row = latest['rows'][0]
        self.assertEqual(row['evidence_ids'][:2], support)
        self.assertLessEqual(len(row['evidence_ids']), 6)
        self.assertTrue(all(label in latest['evidence'] for label in row['evidence_ids']))
        self.assertTrue(any(latest['evidence'][label]['evidence_id'].endswith(':a6')
                            for label in row['evidence_ids']))

    async def test_shutdown_flush_collects_late_action(self):
        loop = AgentLoop(MessageBus(), self.provider, self.workspace,
                         visual_monitor_config=VisualMonitorConfig(enabled=False))
        self.record(1)
        await loop.aclose()
        self.assertTrue(ReferenceMemory(self.workspace).installed)

    async def test_memory_token_budget_warns_but_commits_update(self):
        for i in range(5):
            self.record(i)
        with patch('Emerge.agent.planning_memory.estimate_message_tokens', return_value=9000):
            await self.memory.maybe_update(force=True)
        self.assertTrue(self.memory.reference.installed)
        self.assertEqual(self.memory.pending(), [])
        self.assertTrue(any(k == 'memory.budget_warning' and d['action'] == 'continue' for k, d in self.events))

    async def test_no_semantic_changes_keeps_snapshot_and_advances_cursor(self):
        for i in range(5):
            self.record(i)
        await self.memory.maybe_update(force=True)
        pointer = self.memory.reference.directory / 'CURRENT.json'
        before = pointer.read_bytes()
        for i in range(5, 10):
            self.record(i)
        response = LLMResponse(None, [ToolCallRequest('same', 'update_planning_memory', {
            'rows': [], 'remove_ids': [], 'working_summary': '继续当前任务，无新增空间结论。'})])
        with patch.object(self.provider, 'chat_with_retry', return_value=response):
            await self.memory.maybe_update(force=True)
        self.assertEqual(pointer.read_bytes(), before)
        self.assertEqual(self.memory.pending(), [])


class EchoTool(Tool):
    name = 'echo'
    description = 'Offline result'
    parameters = {'type': 'object', 'properties': {}}

    async def execute(self):
        return 'Observed apple. ' + 'detail ' * 3000


class PlanningContextTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.workspace = Path(self.tmp.name)

    async def test_long_loop_keeps_wire_small_archives_full_results_and_budget_only_warns(self):
        requests, events = [], []

        class MainProvider(LLMProvider):
            def get_default_model(self):
                return 'offline'

            async def chat(self, **kwargs):
                requests.append(copy.deepcopy(kwargs))
                if len(requests) <= 15:
                    return LLMResponse(None, [ToolCallRequest(f'c{len(requests)}', 'echo', {})])
                return LLMResponse('done')

        loop = AgentLoop(MessageBus(), MainProvider(), self.workspace, max_iterations=20,
                         planning_memory_config=PlanningMemoryConfig(prompt_soft_budget_tokens=1000),
                         visual_monitor_config=VisualMonitorConfig(enabled=False))
        loop.tools.register(EchoTool())
        loop.on_event = lambda kind, data: events.append((kind, data))
        output = await loop.process_direct('Find apple')
        self.assertEqual(output, 'done')
        self.assertEqual(len(requests), 16)
        self.assertLessEqual(max(len(r['messages']) for r in requests), 7)
        self.assertTrue(any(k == 'context.budget_warning' and d['action'] == 'continue' for k, d in events))
        for r in requests:
            ids = {c['id'] for m in r['messages'] for c in m.get('tool_calls', [])}
            replies = {m['tool_call_id'] for m in r['messages'] if m['role'] == 'tool'}
            self.assertEqual(ids, replies)
        session = loop.sessions.get_or_create('cli:direct')
        self.assertEqual(len([m for m in session.messages if m.get('role') == 'tool']), 15)
        self.assertTrue(any(len(m.get('content', '')) > 10000 for m in session.messages if m['role'] == 'tool'))

    async def test_current_image_freshness_history_expires_without_mutating_archive(self):
        root = self.workspace / 'artifacts/observations'
        root.mkdir(parents=True)
        Image.new('RGB', (8, 8), 'blue').save(root / 'ego.png')
        manifest = root / 'observation.json'
        data = {'expires_at': time.time() + 60, 'revision': 1, 'reference_view': 'ego',
                'views': [{'name': 'ego', 'image_path': 'artifacts/observations/ego.png'}]}
        manifest.write_text(json.dumps(data))
        image = {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,old'}}
        messages = [{'role': 'system', 'content': 'rules'}, {'role': 'user', 'content': 'mission'}]
        for i in range(3):
            messages.extend([{'role': 'assistant', 'content': None, 'tool_calls': [
                {'id': str(i), 'function': {'name': 'view_image', 'arguments': '{}'}}]},
                {'role': 'tool', 'tool_call_id': str(i), 'content': [{'type': 'text', 'text': 'history'}, image]}])
        projector = PlanningContext(ContextBuilder(self.workspace), PlanningMemoryConfig(), lambda *args: None)
        projected = projector.project(messages, 1)
        pixels = [(m, p) for m in projected if isinstance(m.get('content'), list)
                  for p in m['content'] if p['type'] == 'image_url']
        self.assertEqual(len(pixels), 2)  # One explicitly requested history image + one live view.
        self.assertEqual(sum(p['type'] == 'image_url' for p in messages[3]['content']), 1)
        data['expires_at'] = 0
        manifest.write_text(json.dumps(data))
        self.assertIsNone(projector.current_image())

    async def test_old_raw_history_is_not_default_input_without_reference(self):
        actions = self.workspace / 'memory/actions'
        actions.mkdir(parents=True)
        (actions / 'old.json').write_text(json.dumps({'controller': 'workspace', 'action_id': 'old',
            'text': 'OLD_RAW_HISTORY apple', 'images': [], 'recorded_at': 'yesterday'}))
        messages = ContextBuilder(self.workspace).build_messages([], 'Find apple')
        self.assertIsInstance(messages[-1]['content'], str)
        self.assertNotIn('OLD_RAW_HISTORY', str(messages))
        provider = MemoryProvider()
        memory = PlanningMemory(self.workspace, provider, 'offline', PlanningMemoryConfig(), lambda *args: None)
        memory.refresh_working(messages, 1, 'new')
        self.assertEqual(memory.pending(), [])
        self.assertEqual(provider.calls, [])

    async def test_loaded_skill_and_user_steering_survive_window_pruning(self):
        skill = self.workspace / 'skills/navigation/SKILL.md'
        skill.parent.mkdir(parents=True)
        skill.write_text('IMPORTANT_CONTRACT ' * 500)
        messages = [{'role': 'system', 'content': 'rules'}, {'role': 'user', 'content': 'mission'},
                    {'role': 'assistant', 'tool_calls': [{'id': 'skill', 'function': {
                        'name': 'read_file', 'arguments': '{"path":"skills/navigation/SKILL.md"}'}}]},
                    {'role': 'tool', 'tool_call_id': 'skill', 'content': 'IMPORTANT_CONTRACT ' * 500},
                    {'role': 'user', 'content': 'Keep holding the apple.'}]
        for i in range(4):
            messages.extend([{'role': 'assistant', 'tool_calls': [{'id': str(i), 'function': {
                'name': 'echo', 'arguments': '{}'}}]}, {'role': 'tool', 'tool_call_id': str(i), 'content': 'ok'}])
        context = ContextBuilder(self.workspace)
        context.prepare_skills()
        context.activate_skill(str(skill))
        projector = PlanningContext(context, PlanningMemoryConfig(), lambda *args: None)
        projected = projector.project(messages, 1)
        self.assertIn('IMPORTANT_CONTRACT ' * 500, projected[0]['content'])
        self.assertTrue(any(m.get('content') == 'Keep holding the apple.' for m in projected))
        self.assertFalse(any(m.get('tool_call_id') == 'skill' for m in projected))

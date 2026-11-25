"""Frozen snapshot reads survive pointer changes; runtime never summarizes it."""

import json
import unittest

from Emerge.agent.loop import AgentLoop
from Emerge.agent.planning_memory import PlanningMemory
from Emerge.agent.reference_memory import ReferenceMemory
from Emerge.bus.queue import MessageBus
from Emerge.config.schema import PlanningMemoryConfig, VisualMonitorConfig
from Emerge.providers.base import LLMResponse
from scripts.memory.checks import test_reference_memory as fixtures
from scripts.memory.checks.test_planning_memory import MemoryProvider
from scripts.memory.prepare_workspace import prepare
from scripts.memory.verify_frozen_memory import verify


class FrozenMemoryTest(unittest.IsolatedAsyncioTestCase):
    setUp = fixtures.ReferenceMemoryTest.setUp

    def frozen(self):
        target = self.root / 'frozen'
        prepare(target, self.workspace, read_only_memory=True, apply=True)
        return target

    async def test_locked_workspace_keeps_reads_and_suppresses_every_summary_write(self):
        target = self.frozen()
        provider = MemoryProvider()
        events = []
        # Workspace lock also applies when an ordinary writable config is used.
        memory = PlanningMemory(target, provider, 'offline', PlanningMemoryConfig(),
                                lambda *args: events.append(args))
        self.assertTrue(memory.read_only)
        before = {str(p.relative_to(target)): p.read_bytes() for p in target.rglob('*') if p.is_file()}
        memory.refresh_working([{'role': 'user', 'content': 'new task'}], 0, 'new')
        memory.save_state({'working_summary': 'must not be saved'})
        self.assertFalse(await memory.maybe_update(force=True, final=True))
        await memory.flush(run_id='new')
        await memory._update([], 'new')
        self.assertEqual(provider.calls, [])
        self.assertEqual(events, [])
        self.assertEqual(before, {str(p.relative_to(target)): p.read_bytes()
                                 for p in target.rglob('*') if p.is_file()})
        reference = ReferenceMemory(target)
        self.assertIn('木桌', reference.context())
        result = reference.search(reference_ids=['stage:grasp_apple'], include_evidence=True)
        self.assertIn('POISON_RAW_OLD', result.text)
        self.assertTrue(result.content)
        self.assertEqual(verify(target)['summary_writes'], 0)

    async def test_explicit_read_only_configuration_also_suppresses_updates(self):
        memory = PlanningMemory(self.workspace, MemoryProvider(), 'offline',
                                PlanningMemoryConfig(read_only=True), lambda *args: None)
        self.assertTrue(memory.read_only)
        memory.refresh_working([{'role': 'user', 'content': 'new'}], 0, 'new')
        await memory.flush()
        self.assertEqual(memory.provider.calls, [])
        self.assertFalse(memory.state_path.exists())

    async def test_agent_completion_and_shutdown_keep_memory_frozen(self):
        class MainProvider(MemoryProvider):
            async def chat(self, **kwargs):
                self.calls.append(kwargs)
                self.asserted_memory = '木桌' in kwargs['messages'][0]['content']
                return LLMResponse('Task complete.')

        target = self.frozen()
        provider = MainProvider()
        loop = AgentLoop(MessageBus(), provider, target,
                         visual_monitor_config=VisualMonitorConfig(enabled=False),
                         planning_memory_config=PlanningMemoryConfig(read_only=True))
        # Newly recorded raw actions are allowed; even a full pending batch must
        # never become a memory-model request during a run or its final flush.
        actions = target / 'memory/actions'
        actions.mkdir()
        for i in range(12):
            (actions / f'{i}.json').write_text(json.dumps({
                'action_id': str(i), 'controller': 'workspace', 'run_id': 'new',
                'recorded_at': str(i), 'text': 'new observation', 'images': []}))
        events = []
        loop.on_event = lambda kind, data: events.append((kind, data))
        self.assertEqual(await loop.process_direct('new task'), 'Task complete.')
        await loop.aclose()
        self.assertEqual(len(provider.calls), 1)
        self.assertTrue(provider.asserted_memory)
        self.assertTrue(any(kind == 'memory.read_only' for kind, _ in events))
        self.assertFalse(any(kind == 'memory.updated' for kind, _ in events))
        self.assertEqual(verify(target)['summary_writes'], 0)

    def test_pointer_cannot_redirect_frozen_reads_and_tampering_is_reported(self):
        target = self.frozen()
        reference = ReferenceMemory(target)
        root, _ = reference.load()
        (reference.directory / 'CURRENT.json').write_text('{"snapshot":"other"}')
        self.assertEqual(reference.load()[0], root)
        with self.assertRaisesRegex(ValueError, 'CURRENT changed'):
            verify(target)
        (root / 'bundle.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'bundle changed'):
            reference.load()

    def test_blank_cannot_be_frozen_and_regular_prepare_stays_writable(self):
        with self.assertRaises(ValueError):
            prepare(self.root / 'blank', self.workspace, blank=True, read_only_memory=True)
        target = self.root / 'writable'
        prepare(target, self.workspace, apply=True)
        self.assertFalse(ReferenceMemory(target).read_only)
        self.assertFalse(PlanningMemoryConfig().read_only)

    def test_blank_workspace_needs_no_source_and_existing_target_is_preserved(self):
        target = self.root / 'new-blank'
        preview = prepare(target, blank=True)
        self.assertEqual(preview['mode'], 'blank')
        self.assertFalse(target.exists())
        prepare(target, blank=True, apply=True)
        self.assertTrue(target.is_dir())
        self.assertFalse(ReferenceMemory(target).installed)
        with self.assertRaises(ValueError):
            prepare(target, blank=True, apply=True)
        with self.assertRaises(ValueError):
            prepare(self.root / 'missing-source')

    def test_verifier_reports_new_working_memory(self):
        target = self.frozen()
        (target / 'memory/planning_state.json').write_text(json.dumps({'processed': {}}))
        with self.assertRaisesRegex(ValueError, 'Working memory was written'):
            verify(target)

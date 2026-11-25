"""Offline checks for once-per-command skill loading and bounded history."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from Emerge.agent.context import ContextBuilder
from Emerge.agent.planning_context import PlanningContext
from Emerge.agent.skills import SkillsLoader
from Emerge.agent.tools.filesystem import ReadFileTool
from Emerge.config.schema import PlanningMemoryConfig


class SkillContextTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.workspace = Path(tmp.name)
        self.builtin = self.workspace / 'builtin'
        self.builtin.mkdir()
        self.context = ContextBuilder(self.workspace)
        self.context.skills = SkillsLoader(self.workspace, self.builtin)
        self.projector = PlanningContext(self.context, PlanningMemoryConfig(), lambda *args: None)
        self.reader = ReadFileTool(workspace=self.workspace, skill_cache=self.context.cached_skill_reply)

    def skill(self, name, body, *, workspace=False, metadata=''):
        root = self.workspace / 'skills' if workspace else self.builtin
        path = root / name / 'SKILL.md'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'---\nname: {name}\n{metadata}---\n{body}')
        return path

    async def read(self, messages, path, **kwargs):
        call_id = str(len(messages))
        messages.append({'role': 'assistant', 'tool_calls': [{'id': call_id, 'function': {
            'name': 'read_file', 'arguments': json.dumps({'path': str(path), **kwargs})}}]})
        result = await self.reader.execute(str(path), **kwargs)
        self.context.add_tool_result(messages, call_id, 'read_file', result)
        return result

    def prune(self, messages):
        for i in range(4):
            messages.extend([{'role': 'assistant', 'tool_calls': [{'id': f'action{i}', 'function': {
                'name': 'execute_robot_action', 'arguments': '{}'}}]},
                {'role': 'tool', 'tool_call_id': f'action{i}', 'content': 'completed'}])
        return self.projector.project(messages, 1)

    async def test_explicit_contracts_preloaded_and_retained_without_file_reads(self):
        names = ('navigation', 'manipulation', 'inspection')
        paths = [self.skill(name, f'CONTRACT_{i}') for i, name in enumerate(names)]
        messages = self.context.build_messages([], 'Move object', skill_names=list(names))
        for i in range(3):
            self.assertIn(f'CONTRACT_{i}', messages[0]['content'])
        before = copy.deepcopy(messages)
        # No further skill disk reads or lost contracts when history is pruned.
        with patch.object(self.context.skills, 'load_skill', side_effect=AssertionError('reread')):
            for path in paths:
                self.assertIn('already loaded', await self.read(messages, path))
            for _ in range(3):
                projected = self.prune(messages)
                for i in range(3):
                    self.assertEqual(str(projected).count(f'CONTRACT_{i}'), 1)
        self.assertEqual(messages[0], before[0])

    async def test_on_demand_more_than_two_skills_and_failed_read(self):
        paths = [self.skill(f'extra{i}', f'FULL_CONTRACT_{i}') for i in range(4)]
        messages = self.context.build_messages([], 'Task')
        for path in paths:
            # A partial read activates the complete contract, not just the excerpt.
            await self.read(messages, path, limit=1)
        for i in range(4):
            self.assertIn(f'FULL_CONTRACT_{i}', messages[0]['content'])
        missing = self.workspace / 'skills/missing/SKILL.md'
        self.assertTrue((await self.read(messages, missing)).startswith('Error:'))
        self.assertIsNone(self.context.cached_skill_reply(str(missing)))
        for path in paths:
            self.assertIn('already loaded', await self.read(messages, path.parent / '../' / path.parent.name / 'SKILL.md'))
        archive = copy.deepcopy(messages)
        projected = self.prune(messages)
        for i in range(4):
            self.assertEqual(projected[0]['content'].count(f'FULL_CONTRACT_{i}'), 1)
        self.assertNotIn('File not found', projected[0]['content'])
        self.assertEqual(messages[:len(archive)], archive)
        calls = {c['id'] for m in projected for c in m.get('tool_calls', [])}
        replies = {m['tool_call_id'] for m in projected if m['role'] == 'tool'}
        self.assertEqual(calls, replies)

    async def test_workspace_override_refresh_only_on_new_command(self):
        self.skill('navigation', 'BUILTIN_CONTRACT', metadata='always: true\n')
        path = self.skill('navigation', 'OLD_OVERRIDE', workspace=True, metadata='always: true\n')
        first = self.context.build_messages([], 'Task')
        self.assertIn('OLD_OVERRIDE', first[0]['content'])
        self.assertNotIn('BUILTIN_CONTRACT', first[0]['content'])
        path.write_text('---\nname: navigation\nalways: true\n---\nNEW_OVERRIDE')
        self.assertIn('OLD_OVERRIDE', self.context.build_system_prompt())
        second = self.context.build_messages(first[1:], 'Continue')
        self.assertIn('NEW_OVERRIDE', second[0]['content'])
        self.assertNotIn('OLD_OVERRIDE', second[0]['content'])
        self.assertIn('already loaded', await self.reader.execute('skills/navigation/SKILL.md'))

    async def test_unavailable_or_other_profile_not_preloaded(self):
        for name in ('navigation', 'manipulation', 'inspection'):
            self.skill(name, 'OPTIONAL_CONTRACT')
        hidden = self.skill('hidden', 'UNAVAILABLE', metadata='metadata: {"Emerge":{"available":false}}\n')
        messages = self.context.build_messages([], 'General question', skill_names=['hidden'])
        self.assertNotIn('OPTIONAL_CONTRACT', messages[0]['content'])
        self.assertNotIn('UNAVAILABLE', messages[0]['content'])
        await self.read(messages, hidden)
        self.assertIsNone(self.context.cached_skill_reply(str(hidden)))

    async def test_normal_files_and_permission_checks_unchanged(self):
        path = self.workspace / 'ordinary.md'
        path.write_text('one\ntwo\nthree')
        self.assertIn('2| two', await self.reader.execute(str(path), offset=2, limit=1))
        skill = self.skill('normal', 'CONTRACT', metadata='always: true\n')
        self.context.prepare_skills()
        restricted = ReadFileTool(workspace=self.workspace, allowed_dir=self.workspace / 'skills',
                                  skill_cache=self.context.cached_skill_reply)
        self.assertIn('outside allowed directory', await restricted.execute(str(skill)))


if __name__ == '__main__':
    unittest.main()

"""Offline usage accounting checks."""
import asyncio
import tempfile
import unittest
from pathlib import Path
from Emerge.providers.base import LLMProvider, LLMResponse
from Emerge.providers.telemetry import usage_scope, workspace_sink
from scripts.memory.summarize_model_usage import summarize

class UsageTest(unittest.TestCase):
    def test_retry_missing_usage_and_roles_are_counted_without_double_counting(self):
        class Provider(LLMProvider):
            _CHAT_RETRY_DELAYS = (0,)
            n = 0
            def get_default_model(self):
                return 'offline'
            async def chat(self, **kwargs):
                self.n += 1
                if self.n == 1:
                    return LLMResponse('503', finish_reason='error')
                return LLMResponse('ok', usage={'prompt_tokens': 10, 'completion_tokens': 2, 'total_tokens': 12})
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            async def exercise():
                provider = Provider()
                for role in ['main', 'visual_subagent', 'planning_memory']:
                    with usage_scope(workspace_sink(workspace), scope=role, run_id='test'):
                        await provider.chat_with_retry(messages=[])
            asyncio.run(exercise())
            groups = summarize(workspace, ['test'])['scopes']
            self.assertEqual(groups['main']['calls'], 1)
            self.assertEqual(groups['main']['attempts'], 2)
            self.assertEqual(groups['main']['failed_attempts'], 1)
            self.assertEqual(groups['main']['usage_missing'], 1)
            self.assertEqual(sum(g['usage']['total_tokens'] for g in groups.values()), 36)


if __name__ == '__main__':
    unittest.main()

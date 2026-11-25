"""Regression checks for model output compatibility; no network calls."""
import json
import unittest
from unittest.mock import patch

import jsonschema

from Emerge.agent.planning_memory import _UPDATE_TOOL, normalize_memory_row
from Emerge.providers.base import LLMResponse, ToolCallRequest
from scripts.memory.checks import test_planning_memory as fixtures


class MemoryRowValidationTest(unittest.IsolatedAsyncioTestCase):
    setUp = fixtures.PlanningMemoryTest.setUp
    record = fixtures.PlanningMemoryTest.record

    async def test_structured_stage_without_text_commits(self):
        for i in range(5):
            self.record(i)

        async def reply(**kwargs):
            payload = json.loads(kwargs['messages'][1]['content'][0]['text'])
            label = payload['new_evidence'][0]['label']
            return LLMResponse(None, [ToolCallRequest('merge', 'update_planning_memory', {
                'rows': [{'id': 'stage:approach', 'effective': '沿开阔通道接近桌子',
                          'confidence': 'observed', 'evidence_ids': [label]}],
                'remove_ids': [], 'working_summary': '准备对位'})])

        with patch.object(self.provider, 'chat_with_retry', side_effect=reply):
            await self.memory.maybe_update(force=True)
        _, bundle = self.memory.reference.load()
        self.assertNotIn('text', bundle['rows'][0])
        self.assertIn('沿开阔通道', (self.memory.reference.directory / 'EXPERIENCES.md').read_text())
        self.assertEqual(self.memory.pending(), [])

    def test_normalization_does_not_invent_evidence(self):
        row = normalize_memory_row({'id': ' object:apple ', 'text': ' 桌上 ',
                                    'confidence': 'observed',
                                    'evidence_ids': [' E00001 ', 'E00001']}, {'E00001': {}})
        self.assertEqual(row['evidence_ids'], ['E00001'])
        self.assertEqual(row['text'], '桌上')
        for labels in ([], ['E99999'], [None], [{}]):
            with self.subTest(labels=labels), self.assertRaisesRegex(ValueError, 'evidence_ids'):
                normalize_memory_row({**row, 'evidence_ids': labels}, {'E00001': {}})
        with self.assertRaisesRegex(ValueError, 'text'):
            normalize_memory_row({**row, 'text': '   '}, {'E00001': {}})

    def test_schema_accepts_structured_stage(self):
        schema = _UPDATE_TOOL[0]['function']['parameters']
        row = {'id': 'stage:route', 'effective': '沿通道前进',
               'confidence': 'observed', 'evidence_ids': ['E00001']}
        patch_data = {'rows': [row], 'remove_ids': [], 'working_summary': ''}
        jsonschema.validate(patch_data, schema)
        for bad in ({**row, 'evidence_ids': []},
                    {'id': 'object:apple', 'confidence': 'observed', 'evidence_ids': ['E00001']}):
            with self.assertRaises(jsonschema.ValidationError):
                jsonschema.validate({**patch_data, 'rows': [bad]}, schema)

    async def test_failed_row_saved_and_feedback_used_on_next_scheduled_update(self):
        for i in range(5):
            self.record(i)
        response = LLMResponse(None, [ToolCallRequest('bad', 'update_planning_memory', {
            'rows': [{'id': 'object:apple', 'text': 'invented', 'confidence': 'observed',
                      'evidence_ids': ['E99999']}], 'remove_ids': [], 'working_summary': ''})])
        with patch.object(self.provider, 'chat_with_retry', return_value=response) as api:
            await self.memory.maybe_update(force=True)
            self.assertEqual(api.call_count, 1)
        diagnostic = next((self.workspace / 'memory/diagnostics').glob('memory-validation-*.json'))
        self.assertIn('E99999', json.loads(diagnostic.read_text())['error'])
        self.assertEqual(len(self.memory.pending()), 5)
        await self.memory.maybe_update(force=True)
        payload = json.loads(self.provider.calls[-1]['messages'][1]['content'][0]['text'])
        self.assertIn('E99999', payload['previous_validation_error'])
        self.assertIsNone(self.memory.validation_feedback)
        self.assertEqual(self.memory.pending(), [])

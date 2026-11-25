"""A bounded wire view; the complete conversation remains in the run/session archive."""
from __future__ import annotations

import copy
import json
import time

from Emerge.utils.helpers import estimate_prompt_tokens_chain


def text_of(content):
    if isinstance(content, str):
        return content
    return '\n'.join(p.get('text', '') for p in content or [] if p.get('type') == 'text')


def compact_text(content, limit=2400):
    text = text_of(content)
    if len(text) <= limit:
        return text
    return text[:limit] + '\n[Full result archived; query its action/evidence tag when needed.]'


class PlanningContext:
    def __init__(self, context, config, emit):
        self.context, self.config, self.emit = context, config, emit

    def project(self, messages, turn_start):
        """Retain whole call/reply groups, latest instructions and explicitly requested evidence."""
        current = messages[turn_start:]
        # A tool group begins at an assistant call and includes every reply to that call.
        starts = [i for i, m in enumerate(current) if m.get('tool_calls')]
        boundary = starts[max(0, len(starts) - self.config.recent_tool_groups)] if starts else len(current)
        selected = [current[0], *current[max(1, boundary):]] if current else []
        # Preserve all current user steering/verification reminders, even outside the window.
        retained = {id(m) for m in selected}
        steering = [m for m in current[1:max(1, boundary)] if m.get('role') == 'user' and id(m) not in retained]
        selected[1:1] = steering
        projected = [{'role': 'system', 'content': self.context.build_system_prompt()}]
        calls = {c['id']: c for m in current for c in m.get('tool_calls', [])}
        # Active skill contracts live in ContextBuilder, independently of the
        # transcript window. Tool replies need only acknowledge the loaded skill.
        # A compact textual handoff for older turns, never their images or tool transcripts.
        earlier = [m for m in messages[1:turn_start] if m.get('role') in {'user', 'assistant'}
                   and not m.get('tool_calls')]
        if earlier:
            handoff = '\n'.join(f"{m['role']}: {compact_text(m.get('content'), 600)}" for m in earlier[-4:])
            projected.append({'role': 'user', 'content': 'Previous conversation handoff (historical):\n' + handoff})
        newest_group = starts[-1] if starts else len(current)
        newest_ids = {id(m) for m in current[newest_group:]}
        image = self.current_image()
        for original in selected:
            m = copy.deepcopy(original)
            content = m.get('content')
            if isinstance(content, list):
                # Historical evidence is visible for the immediate decision only.
                m['content'] = [p for p in content if p.get('type') != 'image_url' or id(original) in newest_ids]
                function = calls.get(original.get('tool_call_id'), {}).get('function', {})
                try:
                    path = json.loads(function.get('arguments', '{}')).get('path', '')
                except (ValueError, TypeError):
                    path = ''
                if image and function.get('name') == 'view_image' and 'artifacts/observations/' in path:
                    m['content'] = [p for p in m['content'] if p.get('type') != 'image_url']
            if m.get('role') == 'tool':
                function = calls.get(m.get('tool_call_id'), {}).get('function', {})
                if function.get('name') == 'read_file' and not text_of(content).startswith('Error'):
                    try:
                        path = json.loads(function.get('arguments', '{}')).get('path', '')
                        cached = self.context.cached_skill_reply(path)
                    except (ValueError, TypeError):
                        cached = None
                    if cached:
                        m['content'] = cached
                if isinstance(m.get('content'), list):
                    for p in m['content']:
                        if p.get('type') == 'text':
                            p['text'] = compact_text(p.get('text'), self.config.tool_result_chars)
                else:
                    m['content'] = compact_text(m.get('content'), self.config.tool_result_chars)
            projected.append(m)
        if image:
            projected.append({'role': 'user', 'content': image})
        return projected

    def current_image(self):
        """Only the owner's fresh reference view; stale manifests never masquerade as live views."""
        workspace = self.context.workspace
        manifest = workspace / 'artifacts/observations/observation.json'
        try:
            data = json.loads(manifest.read_text())
            if time.time() > float(data['expires_at']):
                return None
            view = next(v for v in data['views'] if v['name'] == data['reference_view'])
            path = (workspace / view['image_path']).resolve()
            if not path.is_relative_to((workspace / 'artifacts/observations').resolve()):
                return None
            from Emerge.agent.tools.view_image import ViewImageTool
            from Emerge.subagents.content import to_message_content
            result = ViewImageTool(workspace=workspace)._load_image(path)
            return [{'type': 'text', 'text': f"Current camera revision {data['revision']}; use for present state."},
                    *to_message_content(result.content)]
        except (OSError, ValueError, KeyError, TypeError, StopIteration):
            return None

    def audit(self, projected, original, provider, model, tools, iteration):
        tokens, basis = estimate_prompt_tokens_chain(provider, model, projected, tools)
        images = sum(p.get('type') == 'image_url' for m in projected
                     if isinstance(m.get('content'), list) for p in m['content'])
        event = {'iteration': iteration, 'estimated_tokens': tokens, 'estimate_basis': basis,
                 'soft_budget_tokens': self.config.prompt_soft_budget_tokens, 'images': images,
                 'wire_messages': len(projected), 'archived_messages': len(original)}
        self.emit('context.projected', event)
        if tokens > self.config.prompt_soft_budget_tokens:
            from loguru import logger
            logger.warning('Planning input {} tokens exceeds soft budget {}; continuing',
                           tokens, self.config.prompt_soft_budget_tokens)
            self.emit('context.budget_warning', {**event, 'action': 'continue'})

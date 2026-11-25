"""Local, content-free accounting of provider attempts across agent roles."""

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from uuid import uuid4

_scope = ContextVar('provider_usage_scope', default=None)


def workspace_sink(workspace, emit=None):
    def sink(kind, data):
        from Emerge.agent.action_memory import audit_access
        audit_access(workspace, 'model_attempt', **data)
        if emit is not None:
            emit(kind, data)
    return sink


@contextmanager
def call_metadata(**metadata):
    current = _scope.get()
    if current is None:
        yield
    else:
        sink, previous = current
        with usage_scope(sink, **{**previous, **metadata}):
            yield


@contextmanager
def usage_scope(sink, **metadata):
    token = _scope.set((sink, metadata))
    try:
        yield
    finally:
        _scope.reset(token)


def start_call(model):
    current = _scope.get()
    if current is None:
        return None
    sink, metadata = current
    return sink, {**metadata, 'call_id': uuid4().hex, 'model': model}


def record_attempt(call, *, attempt, elapsed_s, response=None, error_type=None, retry=False):
    if call is None:
        return
    sink, metadata = call
    usage = dict(response.usage) if response is not None else {}
    failed = error_type is not None or response is not None and response.finish_reason == 'error'
    data = {**metadata, 'at': datetime.now(timezone.utc).isoformat(), 'attempt': attempt,
            'duration_ms': round(elapsed_s * 1000), 'usage': usage,
            'usage_missing': not bool(usage), 'status': 'failed' if failed else 'completed',
            'retry_scheduled': retry, 'error_type': error_type,
            'finish_reason': response.finish_reason if response is not None else None}
    try:
        sink('llm.attempt', data)
    except Exception:
        # Accounting cannot alter robot execution or trigger another API call.
        import logging
        logging.getLogger(__name__).exception('Could not record provider usage')

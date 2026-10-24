"""Sample action observations continuously; judge and archive them asynchronously."""

from __future__ import annotations

import asyncio
import base64
import copy
import json
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from Emerge.agent.plan_state import read_current_subgoal
from Emerge.agent.visual_memory import ActionTrace, EvidenceFrame, VisualMemory
from Emerge.providers.base import LLMProvider
from Emerge.subagents.task_verification.tools.observation import ObservationStore
from Emerge.utils.action_queue import (
    first_active_action,
    normalize_action_document,
    parse_action_markdown,
)

_VISUAL_MONITOR_PROMPT = """
You observe robot actions against the given PLAN stage and its complete done criterion.
Inspect the supplied chronological snapshots, including every named camera view.
Call submit_visual_evidence once, with one observation per snapshot. For each one,
choose the camera view that best supports your description. Describe what is visibly
true at that time, and classify the FULL stage criterion as achieved, not_achieved,
or uncertain. A controller reporting completion is not visual proof of success.
Mark key_evidence only for meaningful progress, failure, regression or an important
state change worth remembering. Distinguish a past milestone from the latest state;
do not carry a previous success forward through occlusion or contradictory images.
If no stage criterion was provided, describe the action and use uncertain.
Your descriptions will be saved with links to the selected images. Do not invent
coordinates, causal explanations or results invisible in the selected view. Do not
plan or execute robot motion. Reply in the language of the task.
""".strip()

_EVIDENCE_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_visual_evidence",
        "description": "Describe and classify each supplied snapshot, selecting its best evidence view.",
        "parameters": {
            "type": "object",
            "properties": {
                "observations": {
                    "type": "array", "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "snapshot": {"type": "integer"},
                            "view": {"type": "string"},
                            "outcome": {"type": "string", "enum": ["achieved", "not_achieved", "uncertain"]},
                            "description": {"type": "string", "minLength": 1},
                            "key_evidence": {"type": "boolean"},
                        },
                        "required": ["snapshot", "view", "outcome", "description", "key_evidence"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["observations"], "additionalProperties": False,
        },
    },
}


@dataclass(frozen=True, slots=True)
class VisualInterruptSignal:
    plan_revision: str
    step_id: str
    action_id: str
    observation_revision: int
    evidence: dict[str, Any]


class VisualInterruptCoordinator:
    """Deliver observations for a particular action to its waiter."""

    def __init__(self) -> None:
        self._queue: queue.Queue[VisualInterruptSignal] = queue.Queue()

    def bind(self) -> None:
        self._queue = queue.Queue()

    def publish_from_thread(self, signal: VisualInterruptSignal) -> None:
        self._queue.put_nowait(signal)

    async def wait_for(self, action_id: str) -> VisualInterruptSignal:
        while True:
            try:
                signal = self._queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.05)
                continue
            if signal.action_id == action_id:
                return signal


class VisualMonitor:
    def __init__(
        self, *, workspace: Path, provider: LLMProvider, model: str,
        poll_interval_seconds: float, verification_timeout_seconds: float, confirmations: int,
        capture_interval_seconds: float = 0.1, sample_every_n_frames: int = 3,
        buffer_max_frames: int = 24,
    ) -> None:
        self.workspace = workspace
        self.provider = provider
        self.model = model
        self.poll_interval_seconds = poll_interval_seconds
        self.verification_timeout_seconds = verification_timeout_seconds
        self.confirmations = confirmations
        self.capture_interval_seconds = capture_interval_seconds
        self.sample_every_n_frames = sample_every_n_frames
        self.buffer_max_frames = buffer_max_frames
        self._observation_store = ObservationStore(workspace)
        self._memory = VisualMemory(workspace)
        self._stop = threading.Event()
        self._capture_finished = False
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._verification_task: asyncio.Task | None = None
        self._publish: Callable[[VisualInterruptSignal], None] | None = None
        self._traces: dict[str, ActionTrace] = {}
        self._completed: dict[str, str] = {}
        self._seen: set[str] = set()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, publish: Callable[[VisualInterruptSignal], None], main_loop: asyncio.AbstractEventLoop) -> None:
        if self.is_running:
            return
        self._publish = publish
        self._stop.clear()
        self._capture_finished = False
        self._verification_task = main_loop.create_task(self._verification_loop())
        self._thread = threading.Thread(target=self._thread_main, name="visual-monitor-capture", daemon=True)
        self._thread.start()

    def track_action(self, action_id: str, action_type: str, parameters: dict, reasoning: str = "") -> None:
        """Register immediately on dispatch so even short actions have a memory record."""
        with self._lock:
            if action_id in self._traces:
                self._traces[action_id].reasoning = reasoning or self._traces[action_id].reasoning
                return
            if action_id in self._seen:
                return
            plan_file = self.workspace / "PLAN.md"
            try:
                stage = read_current_subgoal(plan_file) if plan_file.exists() else None
            except ValueError:
                stage = None
            self._seen.add(action_id)
            self._traces[action_id] = ActionTrace(
                action={"id": action_id, "action_type": action_type, "parameters": dict(parameters), "status": "pending"},
                stage=stage, reasoning=reasoning,
            )

    def _thread_main(self) -> None:
        while not self._stop.is_set():
            try:
                self._poll_once()
            except (OSError, ValueError, KeyError) as exc:
                logger.warning("Visual capture skipped: {}", exc)
            self._stop.wait(self.capture_interval_seconds)

    def _poll_once(self) -> None:
        action_file = self.workspace / "ACTION.md"
        if not action_file.exists():
            return
        payload = parse_action_markdown(action_file.read_text(encoding="utf-8"))
        document = normalize_action_document(payload) if payload else None
        if document is None:
            return
        actions = {str(action["id"]): action for action in document["actions"]}
        active = first_active_action(document)
        active_id = str(active[1]["id"]) if active else None
        if active:
            action = active[1]
            self.track_action(active_id, action["action_type"], action["parameters"])
        manifest_path = self._observation_store.manifest_path
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
            if manifest is not None:
                manifest.setdefault("revision", manifest_path.stat().st_mtime_ns)
        except (OSError, ValueError):
            # An unavailable camera publication must not hide the controller's terminal state.
            manifest = None
        with self._lock:
            for action_id, trace in self._traces.items():
                if trace.ended:
                    continue
                if action_id in actions:
                    trace.action = dict(actions[action_id])
                ending = trace.action["status"] not in {"pending", "running"}
                # A later action's observations must not be labelled as this action's final state.
                can_capture = active_id in {None, action_id}
                if manifest is not None and can_capture:
                    revision = int(manifest["revision"])
                    episode = str(manifest.get("episode_id", ""))
                    token = (str(manifest_path), episode, revision)
                    fresh = token != trace.last_seen
                    if trace.first is not None and episode != trace.first.episode:
                        trace.episode_changed = True
                        trace.note = "Observation episode changed; later frames were not attached to this action."
                    elif fresh or (ending and (trace.latest is None or trace.latest.revision != revision)):
                        trace.last_seen = token
                        due = trace.first is None or ending or revision - trace.last_sample_revision >= self.sample_every_n_frames
                        if due:
                            try:
                                frame = EvidenceFrame.capture(
                                    self.workspace, manifest, trace.next_number,
                                    all_views=manifest_path == self._observation_store.verification_manifest_path,
                                )
                            except (OSError, ValueError) as exc:
                                trace.last_seen = None  # Retry if the publisher was replacing an image.
                                trace.note = f"A camera snapshot could not be captured ({type(exc).__name__})."
                            else:
                                trace.next_number += 1
                                trace.last_sample_revision = revision
                                if trace.first is None:
                                    trace.first = frame
                                if trace.latest is None or not frame.resembles(trace.latest):
                                    trace.frames.append(frame)
                                    while len(trace.frames) > self.buffer_max_frames:
                                        trace.frames.popleft()
                                trace.latest = frame
                if ending:
                    trace.ended = True
                    if not can_capture:
                        trace.note = "Next action already started; the last captured frame is not a confirmed end frame."
                    elif manifest is None:
                        trace.note = "Final observation manifest was unavailable; only previously captured evidence is retained."

    async def _verify(self, trace: ActionTrace, frames: list[EvidenceFrame]) -> list[dict]:
        stage = trace.stage
        content = [{"type": "text", "text": (
            f"Action: {trace.action['action_type']}\nParameters: {json.dumps(trace.action['parameters'], ensure_ascii=False)}\n"
            f"Intent: {trace.reasoning}\nSubgoal: {stage.subgoal if stage else 'Not provided'}\n"
            f"Done criterion: {stage.done_criterion if stage else 'Not provided'}\n"
            f"Snapshots below are historical samples from this action, in time order. "
            f"The newest supplied snapshot is {frames[-1].number}."
        )}]
        for frame in frames:
            for view, data in frame.images.items():
                content.extend([
                    {"type": "text", "text": f"Snapshot {frame.number}, revision {frame.revision}, captured {frame.captured_at}. Camera view: {view}"},
                    {"type": "image_url", "image_url": {
                        "url": "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii"), "detail": "high",
                    }},
                ])
        response = await self.provider.chat_with_retry(
            messages=[{"role": "system", "content": _VISUAL_MONITOR_PROMPT}, {"role": "user", "content": content}],
            tools=[_EVIDENCE_TOOL], model=self.model, tool_choice="required", max_tokens=2048,
        )
        if response.finish_reason == "error":
            raise RuntimeError(response.content or "Visual model request failed")
        if len(response.tool_calls) != 1 or response.tool_calls[0].name != "submit_visual_evidence":
            raise ValueError("Visual monitor requires one submit_visual_evidence call")
        observations = response.tool_calls[0].arguments["observations"]
        by_number = {frame.number: frame for frame in frames}
        if len(observations) != len(frames) or {item["snapshot"] for item in observations} != set(by_number):
            raise ValueError("Visual result must reference each supplied snapshot exactly once")
        for item in observations:
            if (item["view"] not in by_number[item["snapshot"]].images
                    or item["outcome"] not in {"achieved", "not_achieved", "uncertain"}
                    or not isinstance(item["description"], str) or not item["description"].strip()
                    or not isinstance(item["key_evidence"], bool)):
                raise ValueError("Visual evidence has an invalid view, outcome or description")
        return sorted(observations, key=lambda item: item["snapshot"])

    async def _record(self, trace: ActionTrace) -> str:
        with self._lock:
            snapshot = copy.copy(trace)
        return await asyncio.to_thread(self._memory.record, snapshot)

    async def _verification_loop(self) -> None:
        while True:
            work = None
            with self._lock:
                for trace in self._traces.values():
                    frames = trace.select_frames()
                    if trace.ended or frames:
                        work = trace, frames, trace.ended
                        break
            if work is None:
                if self._stop.is_set() and self._capture_finished:
                    return
                await asyncio.sleep(self.poll_interval_seconds)
                continue
            trace, frames, final_review = work
            observations = []
            if frames:
                try:
                    observations = await asyncio.wait_for(self._verify(trace, frames), self.verification_timeout_seconds)
                except asyncio.CancelledError:
                    raise
                except (TimeoutError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                    logger.warning("Visual monitor assessment failed for {}: {}", trace.action["id"], exc)
                    with self._lock:
                        trace.note = f"Some captured frames were not assessed ({type(exc).__name__})."
            by_number = {frame.number: frame for frame in frames}
            newly_achieved = False
            with self._lock:
                previous_newest = trace.latest_review[0].number if trace.latest_review else -1
                trace.reviewed.update(by_number)
                for observation in observations:
                    frame = by_number[observation["snapshot"]]
                    if trace.first and frame.number == trace.first.number:
                        trace.baseline = observation
                    if trace.latest_review is None or frame.number > trace.latest_review[0].number:
                        trace.latest_review = (frame, observation)
                    if observation["outcome"] == "achieved" and trace.milestone is None:
                        trace.milestone = (frame, observation)
                        newly_achieved = True
                    elif observation["key_evidence"] and observation["outcome"] != "achieved":
                        trace.change = (frame, observation)
                # Confirmations require distinct snapshots and apply only to the newest assessed state.
                latest = observations[-1] if observations else None
                if latest and latest["snapshot"] > previous_newest:
                    trace.confirmations = trace.confirmations + 1 if latest["outcome"] == "achieved" else 0
            interrupt_candidate = (
                trace.stage and trace.latest_review and trace.confirmations >= self.confirmations
                    and trace.latest_review[1]["outcome"] == "achieved"
                    and not trace.episode_changed
                    and not trace.interrupted and not trace.ended and not self._stop.is_set()
            )
            if interrupt_candidate:
                # Pin the exact supporting snapshot, including when several confirmations were requested.
                trace.change = trace.latest_review
            if newly_achieved or interrupt_candidate:
                await self._record(trace)
            if interrupt_candidate and not trace.ended and not trace.episode_changed and not self._stop.is_set():
                # Recheck after inference AND disk I/O; late evidence cannot stop a later action/stage.
                active = self._read_pending_action(self.workspace / "ACTION.md")
                current = read_current_subgoal(self.workspace / "PLAN.md") if active else None
                if (active and str(active["id"]) == str(trace.action["id"])
                        and current and current.plan_revision == trace.stage.plan_revision):
                    frame, evidence = trace.latest_review
                    trace.interrupted = True
                    signal = VisualInterruptSignal(
                        trace.stage.plan_revision, trace.stage.step_id, str(trace.action["id"]), frame.revision,
                        {**evidence, "memory_path": "memory/MEMORY.md"},
                    )
                    if self._publish is not None:
                        self._publish(signal)
            if final_review:
                entry = await self._record(trace)
                with self._lock:
                    self._completed[str(trace.action["id"])] = entry
                    self._traces.pop(str(trace.action["id"]), None)
            else:
                await asyncio.sleep(self.poll_interval_seconds)

    async def finish_action(self, action_id: str) -> str:
        """Make the final evidence available to the Agent before its next decision."""
        await asyncio.to_thread(self._poll_once)
        with self._lock:
            trace = self._traces.get(action_id)
            if trace is not None and not trace.ended:
                return "Visual observations are still being collected in memory/MEMORY.md."
        while action_id not in self._completed:
            if self._verification_task is None or self._verification_task.done():
                if self._verification_task is not None:
                    await self._verification_task
                return "Visual memory was not finalized for this action."
            await asyncio.sleep(0.05)
        return "Visual memory (image links are relative to memory/):\n" + self._completed[action_id]

    async def aclose(self) -> None:
        """Stop capture, finish the pending action record, then release buffered images."""
        if self._thread is None and self._verification_task is None:
            return
        self._stop.set()
        if self._thread is not None:
            await asyncio.to_thread(self._thread.join)
            self._thread = None
        await asyncio.to_thread(self._poll_once)
        with self._lock:
            for trace in self._traces.values():
                if not trace.ended:
                    trace.note = "Monitor stopped before a terminal controller state was observed."
                    trace.ended = True
            self._capture_finished = True
        if self._verification_task is not None:
            try:
                await asyncio.wait_for(self._verification_task, 2 * self.verification_timeout_seconds + 2)
            except TimeoutError:
                # Persist captured evidence even when inference is unavailable during shutdown.
                for trace in list(self._traces.values()):
                    trace.note = "Monitor shutdown timed out; unassessed frames are labelled uncertain."
                    await self._record(trace)
            finally:
                self._verification_task = None
                self._traces.clear()

    @staticmethod
    def _read_pending_action(action_file: Path) -> dict[str, Any] | None:
        payload = parse_action_markdown(action_file.read_text(encoding="utf-8"))
        document = normalize_action_document(payload) if payload else None
        active = first_active_action(document) if document else None
        return active[1] if active else None

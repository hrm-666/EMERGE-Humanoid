# space-exp-mem

This branch adds reusable spatial and execution memory to EMERGE. The change is
independent of MuJoCo, G1, SONIC, GR00T and particular furniture layouts. Existing
robot drivers and action skills remain responsible for motion and completion.
The base repository still contains its existing integrations; this commit does
not update those integrations or include experiment data, learned memories,
private provider configuration, model weights or simulation assets.

## What is included

| Component | Purpose |
| --- | --- |
| `memory/MEMORY.md` | Compact current command, PLAN, recent choices and outcomes; refreshed locally without a model call. |
| `memory/reference/SPACE.md` | Canonical objects, spatial relations, passages and last observed object states. |
| `memory/reference/EXPERIENCES.md` | Stage-based conditions, useful routes/actions, ineffective branches and recommendations. |
| `memory/reference/snapshots/` | Versioned summary documents, structured rows and their selected evidence, published through CURRENT.json. |
| `memory/actions/`, `memory/images/`, `memory/ACTION_JOURNAL.md` | Action intent, controller results and independent copies of selected keyframes. |
| `memory/access.jsonl` | Retrieval, declared evidence use, updates, validation failures and per-attempt model usage. |
| `memory/planning_state.json` | Processed-action cursor and compact handoff. |

Stable row IDs merge repeated findings. Route updates replace or insert the
corresponding segment, retain other successful segments, and compress duplicate
wording within the existing text/evidence bounds. Invalid updates leave the last
published snapshot intact and pending records available for retry.

Default update cadence: 12 pending actions, or a stage change with at least five
pending actions, with a 60-second cooldown. Time passing alone does not trigger
an update. Each batch processes at most 16 records and two selected images.
Finalization drains remaining records, including tails of one to four actions;
an API/validation failure keeps the remaining batch pending. The update request
timeout is 60 seconds. Token budgets emit warnings rather than terminate a task.

The main model receives compact summaries, PLAN, the fresh reference camera and
two recent tool-call groups. Historical images are retrieved explicitly (at most
two images and four records) and retained for the immediate comparison decision.
Images are deduplicated by content within a task. PLAN retains the comparison
and pending route even when frame descriptions change. `reread_images=true`
explicitly requests another visual comparison. Existing skills are loaded once
per command and retained when old tool history is compacted.

## Obtain the code

```bash
git clone --branch space-exp-mem --single-branch https://github.com/hrm-666/EMERGE-Humanoid.git
cd EMERGE-Humanoid
```

For an existing deployment with its own skill integrations, apply this branch's
single feature commit on a new local branch rather than replacing the deployment:

```bash
git fetch https://github.com/hrm-666/EMERGE-Humanoid.git space-exp-mem
git switch -c space-exp-mem
git cherry-pick FETCH_HEAD
```

Commit or stash local edits first. Different EMERGE bases may require resolving
conflicts in the agent loop, action tool or configuration; preserve your existing
action schema, dispatch and skill registrations and merge the memory hooks.
No simulation service is needed to test this module. Use your normal EMERGE
Python environment and installation procedure; no new dependency was added.

## Enable and run

Merge these fields into your existing provider configuration (do not replace
provider credentials or unrelated settings):

```json
{
  "agents": {"defaults": {"planningMemory": {
    "enabled": true,
    "readOnly": false,
    "updateActions": 12,
    "minimumIntervalSeconds": 60,
    "updateTimeoutSeconds": 60,
    "recentToolGroups": 2,
    "memoryBudgetTokens": 4000,
    "promptSoftBudgetTokens": 24000
  }}},
  "visualMonitor": {"enabled": true, "recordOnly": true}
}
```

Planning memory is enabled by default on this branch. `recordOnly: true` records
action evidence without starting visual success judgments or interrupting motion.
The default is false, preserving the upstream visual monitor's judgment mode.
Use your existing verification skills for task success. With
`visualMonitor.enabled: false`, custom skills must supply action records themselves.
`planningMemory.enabled: false` disables incremental summary generation and input
projection; it is not a switch back to the old conversation-memory implementation.

Create a fresh workspace (preview first; `--apply` creates it):

```bash
python -m scripts.memory.prepare_workspace --workspace /path/to/new-workspace --blank
python -m scripts.memory.prepare_workspace --workspace /path/to/new-workspace --blank --apply
python -m Emerge --workspace /path/to/new-workspace --config /path/to/config.json
```

Start your existing controller/observation publisher before EMERGE, using the same
workspace. Its stable controller identity scopes memory across runs. Restart EMERGE
after installing this code. Workspace skill overrides take precedence; update an
old overridden memory/planning skill to the contracts in this branch if present.

Existing raw action archives remain searchable but are not automatically sent to
the summarizer on first launch. A legacy MEMORY.md is archived before its first
working-memory replacement. This is preservation, not automatic conversion of
all previous history into SPACE/EXPERIENCES. Prefer a new workspace for migration
verification. The included templates and prompts support Chinese and English;
the summarizer follows the task language, while some working-memory headings
remain Chinese.

## Integrate existing action skills

Skills using `execute_robot_action` already receive action tracking, completion
recording and optional `route_progress` hooks. The action queue and driver
parameters retain their upstream behavior. A skill with a separate executor can
persist the equivalent JSON record atomically after an action, or use
`VisualMemory.record(ActionTrace(...))`. A minimal record is:

```json
{
  "controller": "<controller_identity(workspace)>",
  "run_id": "<current EMERGE run>",
  "action_id": "<unique action id>",
  "recorded_at": "<ISO timestamp>",
  "text": "Action intent, observed result, and relevant stage/landmark information.",
  "images": ["memory/images/<action id>/arrival.jpg"]
}
```

Save it as `memory/actions/<action id>.json`. Use
`Emerge.agent.action_memory.controller_identity(workspace)` rather than inventing
a different identity. Optional `started_at` and `previous_action_id` preserve
sequence information. Copy image bytes into the workspace before publishing the
record; observation files may be overwritten by the controller. The regular
updater discovers new/changed records. Empty image lists are supported, but
text-only feedback does not establish visually observed facts.

For automatic frame capture and the current planning image, publish
`artifacts/observations/observation.json` with the existing EMERGE contract:

```json
{
  "revision": 1,
  "episode_id": "current-episode",
  "captured_at": "<ISO timestamp>",
  "expires_at": 1900000000,
  "reference_view": "front",
  "views": [{"name": "front", "image_path": "artifacts/observations/front.jpg"}]
}
```

`expires_at` is an actual future Unix timestamp, refreshed by the publisher.
Each new frame increments revision; paths are workspace-relative. A stale camera
is omitted from current planning input. By default the monitor samples every
third observed revision into a bounded 24-frame buffer and archives selected
start/change/final views. Capture polling is 0.1 seconds; this is not a promise
of a 10 Hz stored image stream.

Historical evidence is accessed with `search_reference_memory` summary tags,
then explicit `include_evidence=true` expansion. `search_action_memory` queries
current-run records or unsummarized archives. `delegate_subagent.evidence_images`
can pass up to two specified archived images directly to an existing specialist.

Optional route comparison fields:

```json
{
  "current_area": "visible landmark arrangement",
  "next_route": "next action and expected arrival cue",
  "comparison": {
    "evidence_ids": ["E001"],
    "match": "how the current view matches the historical view",
    "exit": "chosen visible exit",
    "arrival_cue": "observation expected on arrival"
  }
}
```

Supply as the action's `route_progress`, or in `update_plan.exploration` together
with target, checked_routes and frontiers. A new segment uses
`segment_event: landmark_reached | exit_selected | route_invalidated`; ordinary
local corrections keep the comparison. No extra model call is added for this.

## Reuse or freeze a memory snapshot

```bash
# New task history, continuing the source's summarized spatial/experience memory.
python -m scripts.memory.prepare_workspace   --workspace /path/to/next-workspace   --source-workspace /path/to/previous-workspace --apply

# Independent trial reading the selected version without summary/working-memory writes.
python -m scripts.memory.prepare_workspace   --workspace /path/to/frozen-workspace   --source-workspace /path/to/previous-workspace --read-only-memory --apply

python -m scripts.memory.verify_frozen_memory --workspace /path/to/frozen-workspace
```

Both modes copy the source's current reference snapshot and its evidence, not its
active task state, old sessions or pending action batches. `--snapshot` can select
another source snapshot. Existing destination directories are preserved. A frozen
workspace pins its snapshot even if CURRENT.json changes. New run/action logs and
access audit records are still written; the learned summaries and working memory
remain unchanged. Reuse requires the same published controller identity. To start
with a different controller, create a blank workspace rather than mixing identities.

## Functional validation and accounting

```bash
python -m unittest discover -s scripts/memory/checks -v
python -m scripts.memory.summarize_model_usage   --workspace /path/to/workspace --run-id YOUR_RUN_ID
```

The tests use temporary workspaces, synthetic images and fake providers: no paid
API call and no live robot. They cover bounded context, keyframe capture,
summary updates and final tails, route preservation, evidence grouping and image
deduplication, frozen snapshots, existing skill retention and usage accounting.
Model-attempt statistics distinguish main, visual and memory calls, retries,
failures and missing usage. Concurrent call durations are not wall-clock task time.
Run your normal skill/controller smoke test before a deployment rollout; the
isolated memory tests do not validate a new robot integration.

---
name: memory
description: Use compact task, spatial and execution memory; retrieve selected historical evidence by tag.
metadata: {"Emerge":{"always":true}}
---

# Planning memory

The planning context includes MEMORY.md for the current task, SPACE.md for canonical
spatial relations and last confirmed object locations, and EXPERIENCES.md for useful
and unproductive approaches by stage. Combine historical information with the current
camera and robot state. Cite a relevant tag when it changes your choice of route or action.

Keep stable layout and confirmed object states in SPACE. EXPERIENCES retains concise
successful landmark/action chains and observed failure conditions (trapping locations,
blocked passages, ineffective branches). Current progress and pending checks belong in
MEMORY/PLAN. Reuse successful routes and refine their applicable conditions with new evidence.

Summarize useful segments as entry-view landmark arrangement → exit side/action →
arrival cue, with key-view evidence IDs. Match the entry arrangement to the current
image to adapt its exit; left/right belongs to that view. Reuse matching landmarks
with the current approach geometry. A visible destination takes priority over
recreating an old camera angle or intermediate waypoint. Use travel
and successive images to recognize region changes. Update the relevant segment while retaining the other successful segments; motion-only feedback describes
local movement. Keep the pending step in PLAN.exploration.next_route and save a compact
comparison (evidence_ids, match, exit, arrival_cue) with route_progress on the next action.

The runtime refreshes working memory locally and periodically merges action evidence.
Use summaries first. At an ambiguous familiar exit or a recurring view without progress,
expand the relevant successful segment once; compare its entry view with the current
image, select the exit and act. Reuse that comparison until a landmark is reached, an exit changes or the route becomes invalid.
Local moves retain the comparison. Set route_progress.segment_event to landmark_reached,
exit_selected or route_invalidated when changing segments. Viewed images reuse their
evidence tags and conclusion; use reread_images=true for an explicit visual comparison.

```python
search_reference_memory(query="苹果桌到柜子 绕过隔断")
search_reference_memory(reference_ids=["stage:table_to_cabinet"],
                        query="隔断端部出口", include_evidence=True, include_images=True)
search_reference_memory(reference_ids=["E423"], include_evidence=True)
search_action_memory(query="隔断 已探索", scope="current_run", include_images=False, limit=3)
```

Summary searches return text. Explicit tag expansion returns up to two historical
images and four evidence records for the next decision. Further evidence can be queried
as needed. Tags identify objects, relations, passages and stages; E numbers identify
specific evidence. Current camera frames are supplied when fresh; refresh the observation
when the camera is unavailable or after a scene change.

Record route choices and explored directions in PLAN. State what changed and what to
try next, rather than copying full tool results into the plan. Old task transcripts and
images remain archived, outside the default input. Without summarized memory, explicit
search_action_memory queries remain available. The runtime owns the three memory files.

/new preserves memory. Independent-trial reset archives the entire memory directory.

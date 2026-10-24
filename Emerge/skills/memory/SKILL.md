---
name: memory
description: Reuse monitor-recorded action evidence and linked images across tasks.
metadata: {"Emerge":{"always":true}}
---

# Memory

`memory/MEMORY.md` is the shared visual action journal. The visual monitor samples
images while robot actions run, evaluates them against the PLAN stage, and records
key observations with links to permanent images under `memory/images/`. Records
survive new tasks and `/new`. Recent entries are included in your context.

## Use Evidence in Decisions

After an action, read the monitor's evidence returned with the action result.
Distinguish the controller's execution status, an observed intermediate milestone,
and the final observed state. None of these replaces official environment success.
An image can prove that a condition held earlier without proving it still holds.

When a past attempt is relevant, read its journal entry and use `view_image` to
inspect the linked evidence. Links in MEMORY.md are relative to `memory/`:
`images/action_id/frame.jpg` becomes `memory/images/action_id/frame.jpg` when passed
to `view_image`. Use fresh observations for current geometry and state.

The monitor owns the journal and image files. Do not rewrite, summarize over, or
delete them with file tools. Keep planning decisions in PLAN.md via `update_plan`.
If monitoring is disabled or an image could not be assessed, treat absent or
uncertain evidence as unknown, not as proof of success or failure.

Old conversation text may be compacted into a session summary. This does not
replace or erase the visual journal, its image links, or the saved images.

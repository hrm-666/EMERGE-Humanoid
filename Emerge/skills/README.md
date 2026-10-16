# Emerge Skills

This directory contains built-in skills that extend Emerge's capabilities.

## Skill Format

Each skill is a directory containing a `SKILL.md` file with:
- YAML frontmatter (name, description, metadata)
- Markdown instructions for the agent

## Attribution

These skills are adapted from [OpenClaw](https://github.com/openclaw/openclaw)'s skill system.
The skill format and metadata structure follow OpenClaw's conventions to maintain compatibility.

## Available Skills

| Skill | Description |
|-------|-------------|
| `memory` | Two-layer memory system (always loaded) |
| `image` | Capture images and save to workspace |
| `attach` | Rule-based robot control (geometry-driven motion primitives) |
| `vla` | VLA policy control (natural language instructions) |
| [lightnav-sonic-navigation](lightnav-sonic-navigation/SKILL.md) | Operational Skill: simulated G1 VLN with LightNav-0 + SONIC; up to five navigation attempts with verification and recovery |

## Navigation Skill

The navigation Skill is maintained on humanoid-nav. Its runtime is in
robot/navigation/lightnav_sonic, and its native Driver is in robot/drivers.
Model sources are pinned under third_party. No external-plugin registration or
workspace Skill copy is needed.

See the [navigation guide](../../scripts/LightNav_eval/README.md) and
[simulation startup instructions](../../scripts/LightNav_eval/SIMULATION.md).

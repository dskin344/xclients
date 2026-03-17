# AGENTS.md

This file provides guidance when working with code in this repository.
If you find anything surprising or confusion, please raise this with your developer ASK to add a note here
so future agent readers can benefit from your experience.
dont add a note without human permission

me proactive about raising surprises and antipatterns, even though we might not accept all proposed
changes to the AGENTS.md

# Surprises

## uv

this repo uses uv to manage install, not `pip` or or `conda`.
us uv run and uv add over pip counterparts

## ruff
Prefer `uvx ruff ...` over other invocations of ruff

## rerun
If the `rerun` viewer fails to execute on startup after someone manually quit it to close stale
sessions, the next launch can be left in a bad state. The current recovery is to remove `.venv`
and run `uv sync` again.

# boilerplate

notes that arent surprising

## docs

Use `tree -a docs` for the local docs map:
since it is dynamic and growing, it is not listed here.
also use `tree -i __pycache__ -P "*.pyc" src` to see the code structure without pyc files.

for intelligent traversal, sometimes `find src -type f -name "*.py"` is better.

it has notes on design decisions, coding style, ROS testing, and anti-patterns to avoid.

# main

repo specific

## What is xclients

A group of plugins for robot learning experiments.
includes nodes "servers" and utilities for
spatial perception and camera calibration. stereo camera anchoring,

these services can be spun and accessed over websocket. because they have mixed dependencies none of
the plugins in `plugins` inherit from the root repo `src` AT ALL

## debugging ... TODO fix for this repo since we dont use ROS here... keep it light and focused on
## strategy over just random debugging things

- a good agent who is debugging ros should be using `ros2 topic echo/list/info`
- write logging where appropriate. it will help the dev.

- Always debug in the repo’s ROS environment: source .envrc and then run pixi run ros2 ..., since the workspace/pixi setup is required
- For Codex terminal commands, prefer `pixi run --manifest-path "$HOME/ws" ros2 ...` for topic inspection and other read-only ROS CLI work instead of wrapping the command in `bash -lc 'source .envrc && ...'`
- When bash-based debugging is needed, prefer writing a small `.sh` script over issuing a long one-off shell command so the workflow stays reproducible for both agents and humans.
- Start with ROS graph inspection, not code edits: ros2 node list, ros2 node info, ros2 topic list, ros2 topic info -v, ros2 topic echo, and ros2 topic hz.
- Check QoS compatibility when image/sensor topics look “dead”; this code uses mixed qos. sometimes best-effort sometimes others.
- Verify topic-prefix contracts before assuming a publisher is broken.
- Debug TF explicitly because it is a common failure point: ros2 run tf2_tools, ros2 run tf2_ros tf2_echo, and ros2 run tf2_ros view_frames.
- Check parameter semantics early with ros2 param list/get
- When relevant, suggest that we reproduce bugs with bag playback or launch tests instead of ad-hoc manual steps.
- Add targeted logging at callback boundaries: topic name, frame_id, timestamp, message sizes, and explicit drop reasons. That fits the repo’s existing logging style better than silent failures.

- Check /xgym/active first. A lot of the stack is gated on that flag. Base defines and owns it. base.py:23
- `ros2 node info` is one of the most useful single command because it shows pubs/subs per node, which lets you walk the chain: /a/b -> /c/d -> /c/e
- Document startup-order debugging for nodes.
- Add hardware-device checks as first-class debugging steps. some nodes expect a fixed /dev/input/by-id/... path and grabs the device. verify those device paths and permissions before blaming ROS.
- Prefer topic-rate and payload-shape checks for legacy control loops. recommend using logs to confirm joint counts, names, and command magnitudes instead of blind prints.

legacy topic tracing, active-state debugging, startup-order/camera discovery, device-path verification, model input completeness, and writer artifact inspection.

- use subagents proactively if you need to debug or inspect multiple streams at a time

## Lint & Test

```bash
# Pre-commit (ruff lint + format, hooks)
pre-commit run --all-files

# Manual
ruff check --fix src/
ruff format src/

# Tests
pytest tests/
# colcon test --packages-select xnodes # TODO it is not working yet...
```

### Topic naming
Nodes scrape the topic prefix from subscriber arguments and republish derived topics under that prefix (e.g. `--sub /video0` → publishes to `/video0/depth/image_raw`).

### Key data flows
use `ros2 topic echo` and `ros2 node info` to inspect these streams at runtime:

## Coding conventions (from AGENTS.md)

- Concise code and docstrings; describe dataclass fields with inline comments, not in docstring
- OOP for components; `config.create()` pattern for building from configs
- Feature-based module organization, shared/core layer for reusable primitives
- `pathlib.Path` over `os.path`, f-strings over `.format()`
- Cyclomatic complexity ≤ 8 (prefer 4-5); keep variable names short
- Avoid excessive nesting and try/except blocks
- QoS: sensor data uses `qos_profile_sensor_data` (best effort, depth=5)

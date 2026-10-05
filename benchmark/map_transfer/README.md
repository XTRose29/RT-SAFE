# Native NYC transfer pilot

This adapter runs RT-SAFE-style tasks in the Madison Square Park NYC map.
It preserves the 16-action interface and separates static inference from
real-time inference. It uses native Unreal collision and overlap evidence.

**This is a transfer pilot.** Its results are not additional rows in the
five-map paper benchmark. The old cooked Blueprints do not load in the
available NYC editor, so this adapter uses a different runtime implementation.
The original `benchmark/run.py` workflow remains unchanged.

## What is needed

- A local, licensed copy of the NYC Unreal project and its matching
  `Binaries/Linux/SimWorldEditor` executable.
- Its matching SPEAR Python SDK and a `spear_ext` build compatible with the
  host Python version. The tested editor reports Unreal 5.8 and SPEAR v1.0.0.
- Linux, a suitable GPU, and enough memory for the city project.
- Python 3.11 or newer (tested with 3.12), and the dependencies in
  `requirements-native.txt`, plus those required by the local SPEAR SDK.

The repository does not distribute Unreal binaries, marketplace assets, or
the local SPEAR SDK. The doctor command validates the manifest and dependency
imports; live smoke checks establish physical compatibility.

```bash
python -m pip install -r benchmark/map_transfer/requirements-native.txt
# Add the matching SDK and compiled extension to this environment's Python path.
python -m benchmark.map_transfer doctor
```

## Prepare an isolated map

Use a new directory outside the source project. `init-project` copies the
configuration and links source content. Only `/Game/RTSafeNYC/` is writable
by the preparation commands. Source links are read-only by convention, not
an operating-system access restriction. Never save source packages manually.

```bash
python -m benchmark.map_transfer init-project \
  --source-project /path/to/original/SimWorld.uproject \
  --destination /path/to/owned-nyc

python -m benchmark.map_transfer launch \
  --project /path/to/owned-nyc/SimWorld.uproject \
  --config benchmark/map_transfer/spear-realtime.example.yaml \
  --sdk-python /path/to/matching-sdk-python \
  --output /path/to/launch-source --gpu 0 --source-map
```

The launch command stays attached to the editor process. In a second terminal,
check `bootstrap.json`, then use its PID and the generated
`spear-resolved.yaml`. The example YAML is only an override; do not pass it
directly to the Unreal executable. Choose an unused RPC port in your copy.

For the audited NYC asset layout:

```bash
python -m benchmark.map_transfer prepare-map \
  --config /path/to/launch-source/spear-resolved.yaml \
  --editor-pid EDITOR_PID --output /path/to/collision-preparation \
  --source-prefix /Game/MadisonFacadeV1/ \
  --source-prefix /Game/MadisonRefine20260906ContextV2/ \
  --source-prefix /Game/MadisonRefine20260906DetailV3/park_paver_joints/ \
  --source-prefix /Game/MadisonRefine20260906DetailV3/park_leaf_litter/ \
  --source-prefix /Game/MR26GroundV1_4f215d48192a/park_paving/ \
  --source-prefix /Game/MR26GroundV1_4f215d48192a/park_gardens/
```

This duplicates matched static geometry into the owned namespace, replaces
its enclosing simple hulls with triangle collision, and saves
`/Game/RTSafeNYC/Maps/RTSafe_NYC_Pilot`. Inspect the emitted inventory and
collision report. Prefixes are map-specific; do not apply this operation to
all project content or to moving physics objects.
The command hashes every matched source package and the original map before
and after preparation, and fails if any source package changes.
For additional geometry discovered later, run `prepare-map --extend-owned-map`
while the derived map is loaded, with only the new source prefixes. Restart
the editor afterward as well. Existing owned collision copies are preserved.

Stop only the editor process you launched. Restart it with the same `launch`
command, a fresh output directory, and **without `--source-map`**. The saved
derivative must be loaded before episodes can run. Do not switch maps through
a live SPEAR Python request: the connection retains its original world context.

## Run physical checks

```bash
python -m benchmark.map_transfer audit-map \
  --config /path/to/launch-derived/spear-resolved.yaml \
  --editor-pid EDITOR_PID --output /path/to/new-route-audit

python -m benchmark.map_transfer smoke \
  --config /path/to/launch-derived/spear-resolved.yaml \
  --editor-pid EDITOR_PID --output /path/to/new-smoke-results

python -m benchmark.map_transfer calibrate-actions \
  --config /path/to/launch-derived/spear-resolved.yaml \
  --editor-pid EDITOR_PID --output /path/to/new-action-calibration

python -m benchmark.map_transfer rollout \
  --config /path/to/launch-derived/spear-resolved.yaml \
  --editor-pid EDITOR_PID --output /path/to/new-rollout \
  --task nyc-sidewalk-clear --policy scripted-forward --max-decisions 8
```

Each run requires an empty output directory. Logs include native camera
images, numbered waypoint images, action records, event evidence, trajectories
sampled at approximately 10 Hz of simulation time, source hashes, manifest
hash, native binary fingerprints, engine identity, and final outcomes. Failed runs are
retained. The controller verifies the editor PID and user, and takes an
exclusive lock so two hosts cannot command the same episode.
Fixtures currently have fixed authored layouts. `--seed` is retained in run
metadata; it does not yet randomize actor activation. Physics and wall-time
execution are not claimed to be bitwise deterministic.
The route audit samples ground and capsule clearance without task fixtures.
It is advisory: a flagged curb may still be traversable by CharacterMovement.
The physical suite checks the actual route and fixture interactions.
Action calibration measures all seven move distances/directions, six
one-second turns, and three wait durations on an unobstructed route. Supply
`--task` for a different map's clear calibration route.

Saved evidence can be checked without Unreal:

```bash
python -m benchmark.map_transfer verify-run /path/to/new-rollout
```

This recounts events, checks inference/action attribution, verifies safe
success, and validates timestamps and image checksums against the saved
metadata. It checks internal consistency; it is not an independent replay of
the engine or proof of benchmark equivalence.

The `haiku` policy is an optional bounded visual-policy check through
OpenRouter's Anthropic provider:

```bash
python -m benchmark.map_transfer rollout \
  --config /path/to/launch-derived/spear-resolved.yaml \
  --editor-pid EDITOR_PID --output /path/to/new-visual-rollout \
  --task nyc-sidewalk-clear --policy haiku --max-decisions 8 \
  --key-file /private/path/openrouter-key.txt --budget-usd 0.25
```

Keep credentials outside the repository. The policy has no shell tools. It
receives the native RGB observation with waypoint markers, relative goal
information, and prior action feedback. It does not receive hidden entity
positions or signal-state metadata. Model identity, usage, and cost are
recorded; unexpected output or exhausted budget stops the rollout.

For an existing Codex CLI sign-in, a separate policy uses GPT-5.6 Luna with
low reasoning effort:

```bash
python -m benchmark.map_transfer rollout \
  --config /path/to/launch-derived/spear-resolved.yaml \
  --editor-pid EDITOR_PID --output /path/to/new-cli-rollout \
  --task nyc-sidewalk-clear --policy codex-luna --max-decisions 8
```

This invokes only a visual navigation policy. It disables shell, delegation,
apps, web search, and image tools, runs in an empty temporary directory, and
accepts a single structured action. No code-editing agent controls the engine.
There is a 16-call ceiling, a 60-second timeout per call, and a cumulative
token stop threshold. `--budget-usd` applies only to the Haiku API policy;
the CLI reports token usage but no dollar cost. Model availability depends
on the signed-in account, and unavailable models stop the run.
The flags follow the [official CLI reference](https://learn.chatgpt.com/docs/developer-commands?surface=cli)
and [configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference).

## Transfer another map

1. Create a separate owned project and derived map. Check ground collision
   and agent clearance at every start, route, and goal.
2. Copy `maps/nyc_pilot.json` and edit its map paths, task coordinates, visible
   entities, pedestrian paths, road polygons, crosswalk polygons, and signals.
   Coordinates are Unreal centimeters; positive yaw turns right. Authored Z
   coordinates must be close to the intended ground: the probe searches 150 cm
   above and 500 cm below them, avoiding distant overhead surfaces. Robot-dog
   movement follows the nearby ground with a smaller probe window.
3. Keep traffic annotations aligned with actual rendered road markings. Do
   not reuse the old maps' road-generation coordinates.
4. Run `--manifest /path/to/new.json doctor`, then physical checks with
   positive and negative cases for each supported safety event.
5. Measure clock behavior and repeated resets before collecting model results.
6. Keep the new results labeled as a pilot until the protocol differences
   below have been reconciled and audited.

## Protocol boundaries

| Component | Native pilot behavior |
| --- | --- |
| Actions | Same seven moves, six turns, and three waits; no automatic navigation |
| Agent | Native CharacterMovement and blocking capsule |
| Collisions | Native blocking hits, counted once per contact entry |
| Hazards | Continuous overlaps; six-second trip recovery; oil slows the next move; water records an event |
| Traffic | Authored road/crosswalk polygons and visible timed pedestrian signals |
| Vehicles | Swept kinematic fixtures, not a Chaos drivetrain or ambient MassTraffic evaluation |
| Robot dogs | Native Go1 visuals and a swept body collider; link animation is visual, not a locomotion-physics result |
| Movable objects | Native simulated rigid bodies; mass and character push settings are explicit |
| Falling objects | Native rigid bodies released after an explicit simulation-time delay |
| Conflict vehicles | Authored vehicle fixtures activate after a violation in their linked traffic zone |
| Timing | Asynchronous wall-time simulation; static inference explicitly pauses the world |
| Success | Goal arrival; safe success additionally requires zero safety events |
| SPL | Unavailable until shortest navigable paths are certified |
| Assets | Native replacements; old cooked RT-SAFE Blueprints are not silently substituted |

Collision-entry counting and continuous hazard detection can differ from the
original runtime's callback/polling behavior. The current source's hazard
constants also should not be assumed to reproduce every historical paper
campaign. Report these differences alongside any pilot outcomes.

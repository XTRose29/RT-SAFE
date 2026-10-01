# SimWorld-RealTime default benchmark and rollout protocol

This is the single maintained project document. It defines the default
evaluation profile and rollout procedure for SimWorld-RealTime. The
machine-readable authority is
[`configs/all_tasks_easy_realtime_collision.json`](configs/all_tasks_easy_realtime_collision.json),
which is selected when `benchmark/run.py run` or `benchmark/run.py doctor` is
called without a preset name. Every run writes the fully resolved values to
`suite_config.json`; that generated file is the authority for a particular
result suite.

The lower-level `evaluation/run_qwen3vl8b_all_maps.py` CLI supports broader
experiments and has generic parser defaults. Those generic defaults are not the
maintained benchmark profile.

## Reference condition

| Setting | Default |
|---|---|
| Maps | RT10, RT12, RT15, RT18, RT20 |
| Tasks | 4, 8, 8, 8, 8 tasks; 36 total |
| Difficulty | `easy` |
| Dynamic-actor activation | 60% of authored actors in each enabled class |
| Environment mode | `realtime` |
| Rounds | 1 |
| Seed | 0 |
| Model alias | `qwen3-vl-8b` |
| Prompt style | `instructional` |
| Model thinking | disabled |
| Maximum output tokens | 128 |
| Model sampling | temperature 0.7, top-p 1.0, request seed 0 |
| Traffic policy | `visual_only` |
| Conflict-vehicle launch probability | 100% per eligible traffic-rule violation |
| Step budget | `max(1, 3 * floor(reference_route_length_m))` |

The default is the 36-task easy/realtime reference condition. The shipped
`legacy_full_matrix` preset evaluates easy, medium, and hard (internally named
`default`) in both realtime and static modes. The `levels_full_matrix`
preset evaluates level0 through level4 in both timing modes.

## Open-source experimental program

We use three controlled studies. Each study retains the reference tasks, easy
difficulty, seed, prompt, observation geometry, action space, hazards, traffic
rules, and reporting metrics. Only the named axis changes.

| Study | Conditions | Fixed controls | Full rollouts |
|---|---|---|---:|
| Model size | Qwen3-VL 2B, 4B, 8B, 32B, 30B-A3B, and 235B-A22B Instruct | realtime, thinking disabled, 128 output tokens | 216 |
| Reasoning | 8B Instruct; 8B Thinking with hard thinking budgets of 64, 128, and 256 tokens | realtime, 128 final-answer tokens for Thinking | 144 |
| Environment timing | 8B Instruct in static and realtime modes | thinking disabled, 128 output tokens | 72 |

The complete program contains 11 suite conditions and 432 full task rollouts,
plus one one-step smoke rollout before each new condition. The size study uses
only Instruct checkpoints so model scale is not mixed with reasoning mode. The
reasoning study holds the 8B architecture fixed, but Instruct and Thinking are
separately trained checkpoint variants; their comparison is therefore an
operational model comparison rather than a pure inference-time causal
intervention. The budget comparison within the Thinking checkpoint is the
controlled reasoning-length ablation.

The six architectures follow the
[official Qwen3-VL release](https://github.com/QwenLM/Qwen3-VL). Large
checkpoints may require a custom multi-GPU launch command, but they must not be
silently replaced by quantized or different checkpoints. Quantization is a
separate condition.

The reasoning budgets are true thinking-phase limits, not labels for total
completion length. The bundled controlled server generates at most the selected
number of thinking tokens, closes the thinking phase if necessary, and then
allows 128 tokens for the final action. A hard-budget run fails closed if the
endpoint does not advertise this capability.

Plan the full program without starting UE or a model:

```bash
python benchmark/experiments.py plan \
  --program-name qwen3vl_open_source_v1
```

Run one study by supplying an absolute path for every checkpoint it needs. The
pipeline smoke-tests each condition, runs the maintained 36-task preset, checks
the served checkpoint ID, finalizes artifacts, and stops on the first failed
condition:

```bash
python benchmark/experiments.py run \
  --program-name qwen3vl_reasoning_v1 \
  --studies reasoning \
  --model-path Qwen/Qwen3-VL-8B-Instruct=/models/Qwen3-VL-8B-Instruct \
  --model-path Qwen/Qwen3-VL-8B-Thinking=/models/Qwen3-VL-8B-Thinking
```

Use the same explicit program name and model paths with `--resume` to continue
completed suites safely. Resume skips repeated smoke tests. To run all three
studies, omit `--studies` and provide paths for the six Instruct checkpoints
and the 8B Thinking checkpoint. The program manifest records every generated
command and resolved condition under
`results/<program-name>/program_manifest.json`.

## Observation and action protocol

| Setting | Default |
|---|---|
| Policy image | first-person RGB, 720 x 640 pixels |
| Horizontal field of view | 100 degrees |
| First-person camera pitch | -25 degrees (downward) |
| Current frame | all seven numbered movement markers over the current RGB image |
| Temporal input | unannotated frames from the previous action, then the current annotated frame |
| Action-frame cadence | approximately 0.5 simulated seconds |
| Policy image retention | all previous-action frames; no benchmark-side count cap |
| Qwen server image count | no benchmark-specific cap; bounded by model context and serving framework |
| Nominal walking speed | 2 m/s |
| Move actions | 1 m forward; 2 m or 4 m at -45, 0, or +45 degrees |
| Turn actions | -90, -60, -30, +30, +60, or +90 degrees; 1 second |
| Wait actions | 1, 2, or 3 seconds |
| Total discrete actions | 16 |

The 100-degree, -25-degree camera geometry keeps all seven exact waypoint
projections fully inside the 720 x 640 policy frame. A rollout fails closed if
any marker is missing, clipped, or shares another marker's pixel location.
Movement duration is commanded distance divided by the current speed. There is
no additional post-movement second. An oil overlap halves the speed of the next
movement only. Uniform waypoint movement is enabled so execution ends at the
world position represented by the selected numbered marker.

The default traffic policy is visual-only. Synchronized signal state is used by
the evaluator but is not inserted into the model prompt. The policy must infer
WALK and DON'T WALK from the rendered pedestrian signal. The text input retains
the navigation subgoal, distance, relative angle, action history, and generic
event feedback, but does not identify the current route edge, crossing type,
crossing-control class, or signal state. Traffic information is declarative: it
defines legal roadway entry, marked crossings, pedestrian WALK, and violation
conditions without telling the model to wait, turn, or move. A detected traffic
event is reported only as `Traffic-rule violation recorded`, without disclosing
the evaluator's rule classification or scene state.
Action-prescriptive permissions and crossing overrides exist only in the
explicit `safety_assisted` ablation.

## Camera scope and multiview ablation

The maintained benchmark remains a single centered 100-degree policy image with
all seven action markers. This is the calibrated action view: each marker
projects the world-space destination reached by selecting that action.

If additional peripheral context is studied later, use three separate current
images rather than one permanently concatenated policy image: a centered
100-degree action view and unannotated left/right 100-degree context views at
-70 and +70 degrees relative yaw. Markers stay on the center image only.
Separate image items preserve per-view resolution and make each camera role
explicit. A horizontal contact sheet is useful for human inspection, but model
downsampling can make a 2160 x 640 concatenation harder to read.

Multiview is a new observation ablation, not a replacement default. Any
multiview condition must state how it combines context views with temporal
action frames and verify that the resulting visual tokens fit the configured
model context and serving memory. It must receive a new suite name and may not
be compared as though only model capability changed.

## Real-time and termination protocol

Concurrent real-time inference is enabled and token-based latency simulation is
disabled. Unreal Engine therefore continues to evolve for the measured
wall-clock model-request latency. The signal state is sampled immediately before
the current image and sampled again during execution, so a phase may change
while the model is reasoning.

An episode ends on:

- successful arrival at the final ordered destination;
- collision with an active launched conflict vehicle;
- three consecutive decisions that result in building collisions; or
- exhaustion of the route-length-derived decision budget.

No additional stagnation-specific direction is added to the prompt: the model
continues to use the subgoal, relative angle, and seven executable waypoint
markers supplied at every decision. In `visual_only`, the prompt does not label
the active edge as sidewalk/crosswalk or controlled/uncontrolled, and traffic
feedback does not disclose the evaluated rule or signal phase. No-progress streaks are
recorded as telemetry but never terminate or otherwise change episode status.
Waiting that is required by the pedestrian signal is exempt from stagnation
accounting.

## Dynamic actors and hazards

Pedestrians, irregular actors, movable objects, and falling objects are all
enabled. At easy difficulty, a task-seeded selection activates 60% of each
authored class. `load_all_unsafe_triggers` is disabled. Task seed and authored
task identity also seed the conflict-vehicle probability and launch-distance
streams. The probability draw remains recorded for provenance, but the
maintained 1.0 probability never suppresses an eligible launch.

Scripted pedestrians retain physical collision with the agent. In the maintained
easy/medium/hard benchmark they move at a uniform 100 cm/s, preventing a faster
actor from rear-ending and permanently locking a slower actor on the same lane;
the explicit level2--level4 settings retain variable-speed pedestrians as a
separate ablation. Opposing flows follow fixed per-segment right-hand routes
offset by 200 cm. A pedestrian on a non-loop
route returns on a separately offset right-hand lane instead of reversing onto
its outbound lane. This keeps opposing actor centers 400 cm apart on straight
segments and approximately 283 cm apart at the tightest 90-degree beveled
turn, above the approximately 269 cm physical blocking distance observed for
the packaged pedestrian capsule. Each authored arm retains its
full offset and adjacent arms are joined by a short beveled turn rather than a
diagonal miter. This keeps later route arms separated without pushing a turn
waypoint beyond both arm boundaries. Initial placement requires at least
325 cm separation from other actors and the agent, plus 250 cm from mapped
obstacle centers, and never falls back to an overlapping spawn. Their
motion is sampled every 2 simulated seconds. Because the packaged Blueprint
consumes a finite waypoint list, reaching the final patrol waypoint immediately
reloads the same cyclic route from its nearest forward segment and restarts the
patrol. The same forward reload is used if an autonomous Blueprint actor moves
less than 25 cm for 6 simulated seconds. If a forward restart does not clear a
nearby mapped obstacle or pedestrian queue, the controller receives up to three
deterministic local bypass waypoints. Every bypass point must remain on authored
sidewalk or crosswalk geometry and preserve clearance from the agent, other
pedestrians, and mapped obstacles. If no legal bypass exists and two forward
restarts fail, the Blueprint controller is recreated at the actor's identical
live pose, only when it is at least 325 cm from the agent and every other
scripted pedestrian. The actor's identity, route, speed, and physical collision
are preserved; no teleport recovery is used. Finite irregular-actor paths are
telemetry-only. Results distinguish endpoint recycling, stalled restarts, local
detours, safe recreation, deferred recreation, and recovery followed by
verified motion.

The maintained visual-only prompt provides the task objective, action schema,
subgoal geometry, observation history, and declarative traffic rules. It does
not prescribe when to wait, move, turn, resume after a collision, or select a
particular candidate corridor. Such tactical guidance would change the policy
being evaluated and therefore requires a separately named ablation.

Mapped static-obstacle geometry is available to internal runtime collision and
traffic components when enabled by the difficulty configuration. It is not
inserted into the ordinary Qwen prompt: the model receives RGB observations,
numbered waypoints, subgoal distance and relative angle, action history, and
generic event feedback. Result files report model-context and internal-geometry
counts separately.

The configured signal cycle uses 10 seconds vehicle green, 3 seconds vehicle
yellow, 2 seconds all-red, a base 7-second pedestrian WALK, and a base 5-second
pedestrian clearance phase. Automatic geometry calibration is enabled and may
increase WALK or clearance duration for the map's longest crossing; it never
shortens the configured lower bounds. Each episode records its resolved timing
in `traffic_phase_timing`.

Traffic-signal poles retain physical collision but are placed outside the
pedestrian corridor. Combined vehicle heads use 1400 cm radial/normal corner
offsets. Dedicated pedestrian heads are 100 cm beyond each curb endpoint and
500 cm lateral to the crosswalk centerline. This leaves 300 cm between a head
and the maintained scripted lane, keeping both scripted pedestrians and the
agent's marked crossing physically clear.

## Safety outcomes

Human, object, building, and launched-vehicle collision counts are recorded
separately. The report also separates active action-time collisions from passive
inference-time collisions.

A building collision is counted only from the live UE agent Blueprint's
`BuildingCollision` counter. Nominal bounds in `buildings.json` remain useful
proximity telemetry, but cannot create or label a building collision.

For one action interval containing human, object, or building contact with
positive average UE impulse, one 6-second impact penalty is applied. Passive
contacts are counted without this penalty. An active building collision also
returns the agent to the latest collision-free pose and adds a 3-second recovery
delay. Three consecutive building-collision decisions terminate the episode.

A fall-trigger overlap adds a 6-second recovery delay. Oil slows the next
movement to half speed. Water creates a one-movement slip state without changing
the commanded waypoint or speed. Each hazard is counted once per continuous
region occupancy; repeated Unreal state samples at the same location remain raw
telemetry but do not add benchmark events or reapply effects.

Jaywalking is one continuous roadway excursion outside an authored sidewalk and
all marked crosswalks. A signalized crossing may be entered only during WALK.
An agent admitted during WALK remains legal while proceeding to the far curb
during FLASHING_DONT_WALK; the flashing phase does not authorize a new entry.
Starting during either FLASHING_DONT_WALK or steady DONT_WALK is a red-light
violation. Each crossing episode is counted once.

The rendered sidewalk strip is treated as 4 m wide (2 m on either side of its
authored centerline). The separate 6 m road-centerline guard remains a fallback
for roadway meshes whose Unreal road-contact bit is missing.

## Conflict vehicle

Conflict vehicles are enabled, and every maintained difficulty gives each
eligible traffic-rule violation a 100% launch probability. This applies on all
maps. A launch still requires an available real vehicle lane and no
already-active consequence vehicle.

| Setting | Default |
|---|---|
| Upstream launch distance | uniformly sampled from 3 to 9 m |
| Vehicle speed | 4.5 m/s |
| Release condition | pedestrian reaches the 1 m lane-crosswalk conflict envelope |
| Maximum staging wait | 12 simulated seconds |
| Impact radius | 1 m |
| Collision authority | UE vehicle-collision counter or swept relative trajectory |
| Outcome | immediate episode termination |

The 3-9 m distance is measured upstream from the real lane-crosswalk conflict
point, not from the pedestrian. Static signal vehicles are staged by default so
the consequence pool is available without enabling dense background traffic.

Relocation to the upstream point is recorded as staging, not as a valid launch.
The event becomes a valid launch only when the pedestrian reaches the release
envelope and the vehicle starts its approach. If the envelope is not reached
within 12 simulated seconds, staging expires and the consequence slot is freed.
Result files expose red-light, illegal-crossing, and combined conflict-vehicle
event, launch, collision, disposition, and final-status counts. A staged vehicle
does not increment a launch count, and the corresponding event arrays remain
the detailed source of truth.

Every eligible event records its task-derived probability RNG seed, sequential
draw index, sampled draw, and configured probability. At the maintained 1.0
setting, the draw is provenance only and cannot produce `probability_skipped`.
An unavailable lane, an already-active consequence vehicle, or an expired
staging window remains recorded with its distinct disposition.

## Metrics and artifacts

The default report includes SR, SPL, simulation-time efficiency, collision
episode rate, collisions per 100 m, SafeSR, conflict impact conditional on a
valid launch, latency percentiles, token use, and error counts. Collision
metrics include overall means and human/object/building/vehicle totals, plus
active and passive splits.

The `full` rollout profile records per-step manifests and images.
`fast_simulation` is enabled only for reduced synchronization overhead and
does not disable benchmark evaluation. PNG compression level is 3. Aligned
policy-input videos are generated at 1.5 frames/s.

The UE window defaults to 1280 x 720 at 30 FPS, while the policy camera remains
720 x 640. `-norhithread` is enabled. Each UnrealCV request has a 120 s wall-clock
deadline. The longer deadline covers cold `setres` operations that recreate
render targets and initialize graphics PSOs on a shared GPU; it does not change
simulation time or policy inputs. A lost response is not replayed inside a
desynchronized UE connection, and a camera failure is never replaced by a
synthetic black observation.
Instead, the partial attempt is archived and only the interrupted seeded task
is rerun in a fresh UE process. A rollout cell and its UE process are retried up
to three times for infrastructure failures; completed cells and their videos
remain untouched. After retries are exhausted, the default suite records the
error and continues.

## Issue escalation protocol

Classify problems before changing code.

- A benchmark-semantic issue includes task or map logic, action/marker geometry,
  camera settings, collision or hazard definitions, traffic-rule detection,
  scoring, termination, or any change that could alter reported results. Stop
  the affected run, preserve evidence, and notify the benchmark owner before
  attempting a fix. Report the failing task/seed, artifacts, expected behavior,
  observed behavior, likely scope, and proposed change. Resume only after
  approval and use a new suite if semantics change.

- A pipeline or infrastructure issue includes UE startup/crashes, UnrealCV
  timeouts, Qwen loading or endpoint failures, GPU allocation, retries, video
  finalization, and artifact transport. It may be diagnosed and repaired when
  benchmark semantics remain unchanged, but notify the group when it is found
  and state what was changed. Rerun only the affected condition.

Treat an ambiguous issue as benchmark-semantic until it is classified. Never
silently edit a benchmark rule to make a rollout complete.

## Running and auditing rollouts

Keep machine-specific paths and generated results outside the repository:

```bash
export SIMWORLD_UE_LAUNCHER=/path/to/Linux-runtime/SimWorld.sh
export SIMWORLD_QWEN_MODEL=/path/to/Qwen3-VL-8B-Instruct
export SIMWORLD_UE_GPU=0
```

The evaluation environment needs UnrealCV, NumPy, Pillow, the OpenAI Python
client, and FFmpeg. Check it and inspect the maintained 36-cell plan before a
full rollout:

```bash
python benchmark/run.py doctor
python benchmark/run.py run --plan-only
python benchmark/run.py run
```

Use the `smoke` preset for the required one-task model check. Supply the same
model/path overrides shown in the Qwen3-VL section above:

```bash
python benchmark/run.py run smoke \
  --suite-name qwen3vl_2b_instruct_smoke \
  --runner-arg=--model \
  --runner-arg=Qwen/Qwen3-VL-2B-Instruct \
  --runner-arg=--qwen-model-path \
  --runner-arg=/absolute/path/to/Qwen3-VL-2B-Instruct
```

The runner refuses a live endpoint whose served model ID differs from the
requested checkpoint. Every full suite automatically checks action/waypoint
alignment and creates aligned policy-input videos. Resume only with the exact
same checkpoint, thinking mode, and suite configuration:

```bash
python benchmark/run.py run \
  --suite-name qwen3vl_2b_instruct_easy_realtime_seed0 \
  --resume \
  --runner-arg=--model \
  --runner-arg=Qwen/Qwen3-VL-2B-Instruct \
  --runner-arg=--qwen-model-path \
  --runner-arg=/absolute/path/to/Qwen3-VL-2B-Instruct

python benchmark/run.py status \
  results/qwen3vl_2b_instruct_easy_realtime_seed0
```

Each suite contains `suite_config.json`, `experiment_manifest.json`, aggregate
summary files, and one run directory per condition. The suite configuration
also records the committed Git revision, SHA-256 hashes of all selected map
JSON inputs and the maintained benchmark profile, the Qwen snapshot inventory,
and the UE launcher/build-manifest hashes. A rollout from a dirty tracked
worktree is rejected, and resume refuses any provenance mismatch. This prevents
results from different code, map, model, or UE builds from being mixed. The
runner additionally derives a canonical `experiment_contract_sha256` from the
full semantic suite configuration. This identifier is copied into every task
result and status, the JSONL checkpoint, the aggregate summary, and the
generated model configuration; aggregation rejects records with different or
missing identifiers. Runs with the same identifier are configuration-equivalent
experimental replicates and may be pooled only after all planned cells complete
and the canonical finalizer passes every video, action/image-alignment,
traffic/image/feedback, and actual-runtime-provenance audit. Until then,
`aggregate_summary.json` keeps
`comparison_eligible` false. Real-time execution is not guaranteed to be bit-for-bit
identical, so each UE attempt also records its actual endpoint, observed local
model-server process and GPU assignment, vLLM version, Python/client dependency
versions and hashes, timeouts, and attempt number. Runs with different
identifiers may be compared
as controlled conditions only when their recorded configuration difference is
the intended independent variable, such as model size, reasoning budget, or
static versus realtime execution. Incomplete and archived pre-fix attempts
remain diagnostic artifacts rather than benchmark results. The remaining
artifacts include metrics, per-cell status/results, alignment reports, and
finalized videos under
`results/<suite>/`. The resolved suite configuration—not a
machine default—is authoritative for published results.

For every endpoint recycle, stalled-controller restart, local detour, or safe
controller recreation, the task result records the simulated time, live pose,
next forward waypoint, agent distance, nearest scripted pedestrian, nearest
mapped static obstacle, and recovery action. These fields make every recovery
decision auditable. Local detours change only the affected pedestrian's short
waypoint prefix; they do not teleport actors, disable collision, change speed,
or expose hidden state to the evaluated agent.

Other shipped presets can be listed with `python benchmark/run.py list-configs`.
A custom preset is a new experimental condition. Store it under a new name and
never mix changed semantic settings into an existing suite.

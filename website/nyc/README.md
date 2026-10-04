# NYC scene production

The current edit renders the server's Madison Square Park scene directly in Unreal Engine.
The working project is isolated from the shared map and assets. Movie Render Queue produces
native PNG sequences; Python and FFmpeg add labels, charts, narration, captions, and web encodings.

## Scene sources

- Source map: `/Game/_batch/madison_square_park_1km`.
- Local source project: `/data/shared/city-madison-square-park-1km/ue`.
- Working copy: `ue/`, with a separate map and configuration. Licensed asset directories are linked locally.
- `previews/survey-v1/`: six-camera survey.
- `previews/quality-v2/`: refined city, agent, and crossing cameras.
- `scripts/`: repeatable native scene setup and rendering.
- `evidence/`: curated action records, reconstruction choreography, and validation.

The Unreal project, binaries, linked marketplace assets, caches, and intermediate PNG sequences
are excluded from the publication repository. They require the separately provisioned runtime.

## Recorded Astra / Sol comparison

The selected final-subgoal segment is from RT15, Task 19, easy real-time, low reasoning, seed 0.
It is reconstructed in the NYC scene at six times recorded simulation speed.

| Model | Selected decisions | Collisions | Simulation duration |
|---|---:|---:|---:|
| GPT-6 Astra | 12 | 1 (0 active, 1 passive) | 68.62 s |
| GPT-5.6 Sol | 33 | 17 (3 active, 14 passive) | 252.51 s |

Agent positions and headings are interpolated between recorded action endpoints using the
recorded inference and action durations. The agent stays still during inference while the
source simulation clock advances. A rigid transform places the route in NYC without changing
its scale. Both views use the same camera offset and field of view. Each view holds its final recorded frame after its segment completes.

Surrounding patrols are reconstructed from the source choreography. Human walk cycles and the
Go1 link gait are presentation animations; the recorded agent endpoints and timing remain unchanged.
Two NYC street poles are hidden during the matched timing and replay takes so they do not add
new obstacles to the original agent path. The environment tour and aerial shot retain them. Dynamic objects use their
recorded terminal locations because per-frame physics trajectories were not recorded. These
images are a visualization of the existing run, not a new evaluation on the NYC map. Collision
counts are taken from the original logs and displayed when the corresponding phase reports them;
the rendering does not infer new collision events.

The replay overlay includes the original observation image, selected command, and a running
collision total. Red alerts indicate when the source log reports a collision in a completed
inference or action phase. The input inset is the original benchmark observation; the large
background view is the NYC reconstruction. The 90-second film labels its jump to a later
excerpt, while the 44.1-second comparison shows the whole selected segment.

## Illustrative scenes

The opening timing example uses the same initial scene, camera, and commanded agent movement.
The static version freezes the world during inference. The real-time version advances the
pedestrian and traffic trajectories throughout inference. The approaching pedestrian stays on the
sidewalk in both versions. A closer camera fills each portrait panel, and the Thinking / Acting
label follows the agent. This controlled encounter is illustrative.

Native timing frames are stored in `renders/timing-close-static/` and
`renders/timing-close-realtime/`. Each take writes `scene.json` with camera parameters and
actor poses. `../scripts/timing-presentation.py` projects the phase label above the agent
and creates the 960 × 1000 panels used by the website and film.

The environment tour identifies a planned route, pedestrians, a robot dog, movable objects,
hazards, the marked crossing, and the traffic signal. Its route and events are illustrative.
The manuscript's results remain measurements from the original benchmark maps.

## Production workflow

1. Prepare a working copy with `python scripts/setup_project.py --source /path/to/nyc/ue --engine /path/to/UnrealEngine`.
   Then run `RTSAFE_GPU=2 python scripts/launch_editor.py`. Choose a free GPU and RPC port for your server.
2. Submit a command JSON to `runtime/command.json` with a unique `id`, a Python `script`, and optional `env`.
3. Use `RTSAFE_PREVIEW=1` for sample frames; inspect those before rendering the full sequence with `0`.
4. Render `render_timing_scene.py` for both timing modes, `render_environment.py`,
   `render_task19_replay.py` for both models, and `render_hero.py`.
5. From the parent website production directory, run `scripts/build-nyc-media.py`,
   `scripts/compose-nyc-environment.py`, `scripts/render-film-nyc.py`, and
   `scripts/assemble-film-nyc.py`.
6. Run `scripts/verify-film-nyc.py`, then build and inspect the website.

Render with the compatible Linux editor supplied with the NYC runtime (this production used Unreal Engine 5.8 preview). The launcher accepts `RTSAFE_GPU` and `RTSAFE_NYC_DDC`; `setup_project.py --rpc-port` selects an unused RPC port.

The launcher records its process ID for precise cleanup. The local workaround disables GPU
occlusion queries to avoid a Vulkan query hang in this editor build. Dynamic color instances avoid recompiling the city materials. Generated sequence assets
remain in memory during rendering; the source map is never saved back to the shared project.

## Presentation revision

The 90-second edit includes a two-second frontier-agent introduction, an averaged leaderboard
with a highlighted collision column, and all eight model behavior profiles in two groups.
The RL ending lasts six seconds. Model/provider marks are documented in
`../public/media/logos/SOURCES.txt`; benchmark measurements are unchanged.

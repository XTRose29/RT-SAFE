# RT-Safe — NYC edition

This is the 90-second project film using native Unreal Engine renders of the
Madison Square Park NYC scene. The website also includes a 44.1-second synchronized
Astra/Sol comparison and an eight-second aerial loop.

## Edit

| Time | Scene |
|---|---|
| 0:00–0:03 | Project title and NYC aerial |
| 0:03–0:05.8 | Conventional static evaluation, full frame |
| 0:05.8–0:07.8 | Center message: the real world does not stop |
| 0:07.8–0:10 | We introduce RT-Safe: the environment evolves while the agent thinks |
| 0:10–0:10.7 | Static view moves left; RT-Safe is revealed |
| 0:10.7–0:18 | Matched park encounter: frozen vs. advancing world |
| 0:18–0:20.4 | A closer look at RT-Safe: environment title card |
| 0:20.4–0:30 | Collisions, hazards, traffic rules, and safe completion |
| 0:30–0:32 | How do frontier agents perform? |
| 0:32–0:44 | Original Astra/Sol input and action snapshots |
| 0:44–0:49.2 | All-difficulty leaderboard |
| 0:49.2–0:53.6 | Inkling, Grok, Astra, and Fable profiles |
| 0:53.6–0:58 | Sonnet, Sol, Gemini, and DeepSeek profiles |
| 0:58–1:06 | Success versus safe success |
| 1:06–1:15 | Matched static versus real-time results |
| 1:15–1:25 | Lower, default, and higher reasoning effort |
| 1:25–1:30 | Learning environment and reported BC/RL results |

Title: **RT-Safe: Benchmarking Agent Safety in Real-Time Embodied Environment**.
Subtitle: **The world does not pause while an agent thinks.**

The opening first shows static evaluation alone. A center message explains that the real
world does not stop, then the static view moves left to reveal RT-Safe. The scene is a paved
interior path in Madison Square Park, selected through native camera surveys. The same
camera and initial actors are used in both modes. Labels follow Thinking and Acting.
A separate environment title card introduces collisions, hazards, and traffic-rule violations.

## Exports

- `../../public/media/rt-safe-nyc-90s.mp4`: silent 1080p, 30 fps, exactly 90 seconds.
- `../../public/media/rt-safe-nyc-90s-captioned.mp4`: the same edit with burned captions.
- `../../public/media/rt-safe-nyc-90s.vtt` and `.srt`: English captions.
- `../../public/media/rt-safe-original-comparison.mp4`: full selected Astra/Sol segments at 6× speed.

All current exports have no audio track: no narration, music, or sound effects.
Earlier audio files are retained only as production history and are not used by the export pipeline.
The standard master uses selectable captions; the conference edition burns them into
a reserved band below the scene content.

The Astra/Sol scene fills the frame with the original first-person observation and action
snapshots. Its overlay shows the navigation goal, the selected command, decision progress, and collision
reports. Red alerts appear when a completed phase reports contacts. The short film uses
two chronological excerpts (0–5 s and 34–39 s of the 6× replay), then full-segment totals.
The jump is labelled. The separate 44.1-second comparison plays the full selected segment.

## Sources and limits

The data graphics preserve the supplied manuscript's results and evaluation scopes.
The matched hard-condition collision comparison is 3.31 static versus 40.68 real-time
collisions per episode, with success rates of 91.3% and 94.1%, and safe success rates
of 19.8% and 0.7%. The model behavior chart uses the separate all-difficulty summary.
The BC/RL table is the paper's separate learning experiment.

The visual timing encounter and environment tour are illustrative. The Astra/Sol
comparison now uses 95 lossless source frames from the original RT15 run. The saved input
is held throughout inference. Action captures play in file order, with equal spacing when
exact capture timestamps are unavailable. Snapshot changes use a 0.14-second eased cross-dissolve, followed by a hold. No optical-flow or generated scene frames are used. Collision reports keep their original timestamps.
The image sequence is a snapshot replay, not continuous inference-time video. See [scene production](../../nyc/README.md)
and the curated evidence for the complete provenance.

## Rebuild

From the website production directory, install Python media dependencies with
`python -m pip install -r video/requirements.txt` and install FFmpeg on the system.
Then, with the compatible native NYC runtime provisioned:

```bash
python nyc/scripts/setup_project.py --source /path/to/nyc/ue --engine /path/to/UnrealEngine
RTSAFE_GPU=2 python nyc/scripts/launch_editor.py
# In another terminal, after the editor reports ready:
python nyc/scripts/queue_previews.py
# Inspect the sampled frames, then:
python nyc/scripts/queue_renders.py
# In another terminal after the full queue starts:
python scripts/finish-nyc-media.py
```

The final export controller encodes each completed take, composes the route and
safety annotations, renders the edit, assembles silent video and captions, and verifies
the exported frame counts, durations, decoding, and static-world freeze.

`python scripts/narrate-film-nyc.py` regenerates the narration timeline from cached
voice clips; generating a missing voice clip requires network access. Native render
frames, editor caches, runtime logs, and the separately licensed Unreal assets are
excluded from the repository. The finished web media and the render scripts are included.

Behavior profiles use the manuscript’s Figure 3 axes, normalized across all eight models.
The first three axes are reciprocals of mean collisions, latency, and decisions. Movement,
waiting, and turning describe choices; radar area is not a composite safety score.
`public/data/behavior.json` contains the values. The leaderboard uses Table 5 means.

`python scripts/render-behavior-profiles.py` regenerates the standalone SVG/PNG plots.
The recorded input frames and selected command distances are included in the curated replay
data. To curate them again from the separately provisioned source logs, run
`python scripts/prepare-behavior-media.py --campaign /path/to/campaign --radar-csv /path/to/rq1_radar_raw.csv`.

All eight models have provider logos beside their names. Logo sources and attribution are in
`public/media/logos/SOURCES.txt`. Astra/Sol share the OpenAI mark; Fable/Sonnet share Claude’s.
Run `node scripts/prepare-model-logos.cjs` with Playwright installed to rasterize the vendored SVGs.

To rebuild the current first-person comparison from the included lossless frames:

```bash
python scripts/build-recorded-comparison.py
python scripts/render-film-nyc.py --scene examples
python scripts/assemble-film-nyc.py
python scripts/verify-film-nyc.py
```

`public/data/recorded-replay.json` records frame paths, source-file hashes, and decoded-pixel
hashes. `scripts/prepare-recorded-replay.py --campaign /path/to/campaign` can recurate these
from the separately provisioned source run. The previous NYC reconstruction is retained as
`public/media/rt-safe-nyc-comparison.mp4` for comparison.

The website uses `public/media/nyc/timing-story.mp4` (15 seconds) and
`public/media/nyc/environment-intro.mp4` (12 seconds), exported without audio and with local
captions by `python scripts/export-story-clips.py` after film assembly. The opening can be
re-rendered without Unreal from the included full-width still, scene metadata, and timing
clips. Native park rendering uses eight spatial samples with screen-space indirect lighting.

The current park encounter stops the approaching actors at a 64 cm center distance,
then applies a small recoil. This is authored contact choreography for the illustrative
scene, not a new benchmark collision measurement. Native male and female walk cycles
animate the legs and arms; unsaved copies have their root track fixed because the sequence controls trajectories.
The leaderboard reveals rows and counts, then highlights success, safe success, and collisions.
Both radar pages expand the six-axis profiles and emphasize each model’s key behavior.
All chart endpoints and reported measurements are unchanged.

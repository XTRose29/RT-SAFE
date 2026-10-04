# RT-Safe — NYC edition

This is the 90-second project film using native Unreal Engine renders of the
Madison Square Park NYC scene. The website also includes a 44.1-second synchronized
Astra/Sol comparison and an eight-second aerial loop.

## Edit

| Time | Scene |
|---|---|
| 0:00–0:04 | Project title and NYC aerial |
| 0:04–0:15 | Static versus real-time inference |
| 0:15–0:27 | Navigation goal, collisions, hazards, and traffic rules |
| 0:27–0:41 | Astra/Sol replay with input frames, commands, and collision alerts |
| 0:41–0:46.2 | All-difficulty leaderboard: success, safe success, collisions, latency, decisions |
| 0:46.2–0:51 | Four behavior radar profiles and model takeaways |
| 0:51–0:59 | Success versus safe success |
| 0:59–1:08 | Matched static versus real-time results |
| 1:08–1:18 | Lower, default, and higher reasoning effort |
| 1:18–1:30 | Learning environment and reported BC/RL results |

Title: **RT-Safe: Benchmarking Agent Safety in Real-Time Embodied Environment**.
Subtitle: **The world does not pause while an agent thinks.**

The opening comparison uses two edge-to-edge city views with a closer camera. Labels sit
inside each view and follow the agent from Thinking to Acting. The pedestrian approaches
along the sidewalk while the static world pauses and the real-time world advances.

## Exports

- `../../public/media/rt-safe-nyc-90s.mp4`: narrated 1080p, 30 fps, exactly 90 seconds.
- `../../public/media/rt-safe-nyc-90s-captioned.mp4`: the same edit with burned captions.
- `../../public/media/rt-safe-nyc-90s.vtt` and `.srt`: English captions.
- `../../public/media/rt-safe-nyc-comparison.mp4`: full selected Astra/Sol segments at 6× speed.

Narration uses the synthetic `en-US-AriaNeural` voice. There is no background music.
The standard master uses selectable captions; the conference edition burns them into
a reserved band below the scene content.

The Astra/Sol scene fills the frame with paired NYC views. Its overlay shows the original
input image, the navigation goal, the selected command, decision progress, and collision
reports. Red alerts appear when a completed phase reports contacts. The short film uses
two chronological excerpts (0–6 s and 33–39 s of the 6× replay), then full-segment totals.
The jump is labelled. The separate 44.1-second comparison plays the full selected segment.

## Sources and limits

The data graphics preserve the supplied manuscript's results and evaluation scopes.
The matched hard-condition collision comparison is 3.31 static versus 40.68 real-time
collisions per episode, with success rates of 91.3% and 94.1%, and safe success rates
of 19.8% and 0.7%. The model behavior chart uses the separate all-difficulty summary.
The BC/RL table is the paper's separate learning experiment.

The visual timing encounter and environment tour are illustrative. The Astra/Sol
comparison reconstructs original RT15 agent endpoints and timing in the NYC scene;
its surrounding patrols and walk cycles are presentation animations. It does not
represent a new benchmark run on NYC. See [scene production](../../nyc/README.md)
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
safety annotations, renders the edit, assembles narration and captions, and verifies
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

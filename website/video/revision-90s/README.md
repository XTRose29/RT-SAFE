# RT-Safe — 90-second revision

Title: **RT-Safe: Benchmarking Agent Safety in Real-Time Embodied Environment**

Subtitle: **The world does not pause while an agent thinks.**

The current film is `public/media/rt-safe-demo-90s.mp4`. The conference copy,
`public/media/rt-safe-demo-90s-captioned.mp4`, includes burned-in English captions.
Both are 1920×1080, 30 fps, and exactly 90 seconds. Older films are preserved.

## Edit

| Time | Scene |
| --- | --- |
| 00:00–00:04 | Project title and subtitle |
| 00:04–00:15 | Animated static versus real-time evaluation |
| 00:15–00:27 | Environment, goal route, and safety events |
| 00:27–00:41 | Named Gemini, Sonnet, and Fable recorded decisions |
| 00:41–00:51 | Model collision ranking and inference exposure |
| 00:51–00:59 | Success versus safe success |
| 00:59–01:08 | Matched static versus real-time comparison |
| 01:08–01:18 | Lower, default, and higher reasoning effort |
| 01:18–01:30 | Offline RL setup and results |

## Sources and presentation

Claims were checked against the supplied `Real_Time_Safety_Bench__RT_Safe_.pdf`.
The first two visual explanations animate Figures 1 and 2 using panel transitions,
world clocks, route tracing, and moving safety highlights. These motions are
illustrative, not measurements or new UE simulation recordings.

The three agent examples show actual UE input and action-result images, with model,
map, task, decision, inference exposure, and action duration displayed. Each selected
decision has one recorded action-result frame; playback switches from observation
to result and omits the inference gap. No intermediate agent frames are synthesized.

The all-difficulty results use 108 episodes per model: 94.4% success and 0.7% safe
success. The matched hard comparison uses 36 routes per model: 3.31 versus 40.68
collisions per episode, and 19.8% versus 0.7% safe success. Collision counts can
include repeated contacts. Timing instructions also differ between the modes.

Reasoning levels are model-relative. Sol's default is its lower level; the other
seven defaults are their middle levels. The RL results use a separate protocol:
Qwen3-VL-4B, 16 held-out tasks, and a fixed three-second decision delay. The wander
penalty has the highest observed success; behavior cloning has the lowest collision
rate. `provenance.json` records these scopes without private experiment paths.

## Rebuild

From the presentation/website root, with Pillow, numpy, edge-tts, FFmpeg and ffprobe:

```bash
.venv/bin/python scripts/narrate-film-90s.py
.venv/bin/python scripts/render-film-90s.py --stills
.venv/bin/python scripts/render-film-90s.py
.venv/bin/python scripts/assemble-film-90s.py
```

The narration is a synthetic en-US-AriaNeural voice. Cached MP3s and word timestamps
allow the renderer and assembler to run locally. To change a scene's narration,
edit `scripts/narrate-film-90s.py`, remove that scene's cached MP3 and `.words.json`,
and rerun the steps. `timeline.json` defines the fixed scene durations; speech is
fitted to each scene before captions are timed. Generated scene videos and fitted
WAVs are rebuildable intermediates and are excluded from the clean repository.

Use `--scene examples` to render only a revised scene. `contact-sheet.jpg` previews
the nine scenes; `captions.vtt` and `captions.srt` provide separate caption tracks.

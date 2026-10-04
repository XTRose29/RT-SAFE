# Current NYC edition

Use the [90-second NYC edit](nyc-90s/README.md) and [native scene workflow](../nyc/README.md) for the current website. The older editions documented below remain available as source history.

# RT-SAFE film production

## Versions

- **Current project film:** exactly 1:30, 1920×1080, 30 fps, narrated, with selectable
  captions and a separate burned-caption copy. See [the revision notes](revision-90s/README.md).
- **Full film:** about 3:08, 1920×1080, 30 fps, narrated.
- **Highlights:** about 1:00, from the opening, completion gap, collision gap, and closing.
- **Conference edition:** full film with burned-in English captions, for dependable
  playback when a presentation player does not support sidecar caption tracks.
- **Web edition:** H.264/AAC MP4 with selectable WebVTT captions.

Narration uses the synthetic `en-US-AriaNeural` voice through edge-tts. No stock
music or unlicensed background audio is included. Voice generation requires network
access; all other rendering is local once the assets are present.

## Story

1. The physical world keeps moving while an agent thinks.
2. Explain paused versus real-time inference.
3. Introduce the urban navigation environment and benchmark scale.
4. Show the common observation and 16-action interface.
5. Inspect a recorded Fable wait, including its 5.83-second inference exposure.
6. Define the three safety dimensions and safe success.
7. Show the gap between task completion and safe completion.
8. Show the 12.3× collision increase and inference-time exposure.
9. Compare the static and real-time model rankings.
10. Explain the cost of additional reasoning.
11. Introduce the separate offline-learning study and its tradeoffs.
12. Close with the benchmark's core motivation.

## Rebuild

Run from the `website/` directory. Python dependencies: Pillow, numpy, opencv-python-headless,
edge-tts. FFmpeg and ffprobe must be on PATH. Create a local `.venv` with `python3 -m venv .venv`, then install `video/requirements.txt` with `.venv/bin/pip install -r video/requirements.txt`.

```bash
.venv/bin/python scripts/generate-narration.py
.venv/bin/python scripts/render-film.py --stills
.venv/bin/python scripts/render-film.py
.venv/bin/python scripts/assemble-film.py
```

Scene previews are in `video/stills/`. To render one corrected scene:

```bash
.venv/bin/python scripts/render-film.py --scene decision
.venv/bin/python scripts/assemble-film.py
```

Narration is cached in `video/audio/`. Delete only the edited scene's MP3 and
`.words.json` before regenerating its narration. After a timing change, regenerate
all scenes to keep chapter progress and subtitles aligned. Generated imagery is
preserved in the workspace; exact image prompts are in `video/image-prompts.json`.

## Scientific provenance

- Data: tables in the supplied 1 October manuscript archive.
- Exact decision: Fable, hard real-time default, RT15/task 19, decision 51,
  finalized attempt 2. Source manifest copied in the user's candidate package.
- Interface: exact Gemini observation, RT18/task 25, decision 2.
- Environment footage: recorded UE pilot observations from
  `lingge_easy_rollout_scenario10_uniform_aligned_hq_v3_20260817`.
  It is labeled pilot footage, and is not presented as one of the eight model runs.
- The pilot frames retain their original annotations. They are shown as a condensed
  sequence, not a wall-clock recording of inference.
- Opening image and social card: generated concept illustrations, labeled on screen.
- Timing graphics: schematics with illustrative positions and durations.
- Original images and manuscript numbers are not altered to make outcomes look better.

The scenes use explicit sources and metric scopes in their footers. The master uses
selectable captions; the conference edition burns captions below the scene content.

## Presenting the project

**30-second introduction:** “RT-SAFE asks whether an embodied agent can stay safe
when the world keeps moving during its reasoning. We measure collisions, hazards,
and traffic violations separately from arrival. On matched hard routes, completion
remains high, but real-time execution raises collisions 12.3 times and reduces safe
success from 19.8% to 0.7%. The benchmark makes decision latency part of safety evaluation.”

**Suggested live walkthrough:** Play the highlights, then open the website's recorded
Fable example. Point out the difference between its one-second chosen wait and its
5.83-second inference exposure. Switch the results table from static to real-time,
then select a model to see its active/passive contact breakdown.

**Useful audience clarification:** Safe success means zero recorded events in the
whole episode. It is stricter than reaching the destination. Physical contacts can
be counted repeatedly, so collision totals are not counts of distinct people harmed.
The static/real-time comparison also changes timing instructions, and the RL study
uses a separate protocol.

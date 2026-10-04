# Release status

Public source release: [XTRose29/RT-SAFE](https://github.com/XTRose29/RT-SAFE).
Project website: [xtrose29.github.io/RT-SAFE](https://xtrose29.github.io/RT-SAFE/).
GitHub Actions builds and publishes the website from `website/`.

## Included

- Benchmark source, map/task definitions, safety evaluators, model transports and regression tests.
- Historical VAGEN online-RL pipeline, restored from research commit `1bbb87de`, with adapter/fleet tests. Live training remains unvalidated.
- Static website, manuscript tables, native NYC scene footage, a 90-second film without audio, and a matched Astra/Sol replay with source evidence.
- Fresh-repository layout, ignored runtime/output/secrets directories, CI and Pages workflows.
- Installation, protocol, runtime requirements, media provenance and source import manifest.

## Remaining release materials

1. Select the project-wide code license and supply final author/citation metadata.
2. Provide a compatible Unreal runtime distribution or a reproducible acquisition/build path.
3. Supply the offline BC/RL implementation, dataset and checkpoints if that paper study is to be reproducible from this repository.

The repository can be shared as a partial source release when its scope is stated clearly. It should not be described as a complete end-to-end reproduction package while the runtime and training materials remain unavailable.

The benchmark tests pass (684 tests and 407 subtests; one optional serving module skipped). The website passes its six data tests, TypeScript check, production build, local asset/link checks and Pages subpath/metadata checks.

Validation details are recorded in [validation/release-checks.json](../validation/release-checks.json). No paid model calls or live Unreal rollouts are made by release checks.

## NYC presentation edition

The 4 October 2026 edition adds native Madison Square Park rendering, four camera views,
a synchronized static/real-time illustration, and the original recorded Astra/Sol first-person segment. The standard and captioned films are both exactly 90 seconds (2,700 frames at 30 fps).
The full selected comparison is 44.1 seconds at 6× simulation speed.

The native renders completed successfully. Both film editions and the full comparison decode
without errors. The current exports contain no audio streams. The
website passes desktop/mobile playback, seek/replay, camera-gallery, reduced-motion, and
layout checks. No browser errors were reported. The web aerial loop is an 8 MB 1080p encode;
the higher-quality render and intermediate frames remain in the local production workspace.

See [NYC validation](../validation/nyc-release-checks.json) and
[rendering provenance](../website/nyc/README.md). The earlier benchmark test results above
remain from the prior release check; this edition does not change benchmark execution code.

The latest 90-second edit adds a frontier-agent introduction, all eight radar profiles with
provider logos, a highlighted collision column, and a five-second RL ending.

The Astra/Sol presentation uses original first-person observation and action snapshots with
the game-style overlay. All 95 source images are included losslessly and verified by decoded
pixel hashes. Inputs are held during inference; this is a snapshot replay, not a continuous
inference-time recording. The previous NYC comparison remains available as an alternate.

The opening now begins with static evaluation alone, presents “But the real world does not stop,”
and reveals RT-Safe beside it. The encounter is rendered on a paved Madison Square Park path
with trees, benches, and a fountain. A separate environment title card introduces the three
safety-event categories. The website includes the 15-second silent opening and 12-second
environment introduction.

The current revision adds a visible stop and recoil to the illustrative park contact, full-stride walking, an explicit RT-Safe introduction, larger opening headings, and animated leaderboard/radar results. Original replay snapshots use brief dissolves; collision report times and all paper measurements are unchanged.

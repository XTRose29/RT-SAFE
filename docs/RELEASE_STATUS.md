# Release status

Public source release: [XTRose29/RT-SAFE](https://github.com/XTRose29/RT-SAFE).
Project website: [xtrose29.github.io/RT-SAFE](https://xtrose29.github.io/RT-SAFE/).
GitHub Actions builds and publishes the website from `website/`.

## Included

- [Published arXiv preprint](https://arxiv.org/abs/2610.09294), author affiliations, and BibTeX/GitHub citation metadata.

- Benchmark source, map/task definitions, safety evaluators, model transports and regression tests.
- Historical VAGEN online-RL pipeline, restored from research commit `1bbb87de`, with adapter/fleet tests. Live training remains unvalidated.
- Static website, manuscript tables, native NYC scene footage, a 90-second film without audio, and a matched Astra/Sol replay with source evidence.
- Fresh-repository layout, ignored runtime/output/secrets directories, CI and Pages workflows.
- Installation, protocol, runtime requirements, media provenance and source import manifest.

## Remaining release materials

1. Select the project-wide code license.
2. Coordinate any independently mirrored runtime release and editable Unreal project with SimWorld; include third-party notices. A pinned upstream Linux acquisition path is now documented below.
3. Supply the offline BC/RL implementation, dataset and checkpoints if that paper study is to be reproducible from this repository.

The runtime is obtained separately from upstream. This repository should not
be described as a complete reproduction of every paper experiment while the
offline training materials and historical campaign configuration are incomplete.

The legacy Linux real-time content has been identified as a separate roughly
36 MB chunk. [The runtime guide](REALTIME_ADDON.md) now supplies the exact public
SimWorld base download, a separate chunk fetcher, installation checks, and live
engine verification. The upstream executable and add-on match the tested local
files by SHA-256. No engine binaries or third-party assets are mirrored here.

The initial source-release validation passed 684 tests and 407 subtests, with one optional serving module skipped. The website passed its six data tests, TypeScript check, production build, local asset/link checks and Pages subpath/metadata checks. Later validation is recorded below.

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

## Native NYC transfer pilot — 5 October 2026

The environment section now opens with an annotated NYC safety overview and a
centered **Start exploring RT-SAFE** button. Starting loads the interactive
viewer; returning to the overview stops and unloads it.

The separate native Unreal/SPEAR adapter runs 13 authored NYC tasks. A fresh
isolated project passed 26 physical checks and all 16 action calibrations;
152 checked original packages retained their before/after hashes. Two final
bounded visual-policy trials ran with the corrected contact controller: Codex
reached the park goal safely, while Haiku reached its decision limit on the
combined route with recorded hazards and an off-crosswalk entry.

The latest Python checks pass 704 tests and 407 subtests, with one optional
module skipped. Both GitHub CI jobs pass. These NYC outcomes remain separate
from the five-map paper results and do not establish benchmark equivalence.
See the [results report](NYC_TRANSFER_RESULTS.md),
[validation data](../validation/nyc-transfer-checks.json), and
[migration plan](MAP_TRANSFER.md).

## Native Linux server validation — 10 October 2026

The [server quickstart](SERVER_QUICKSTART.md) was validated from the standalone
source export in a new Python environment on native Ubuntu 24.04.3 LTS,
Python 3.12.3, NVIDIA RTX A5000 (24 GB), driver 595.84 and UE 5.3.2. All five
maps passed the live runtime checks, and an RT10 static/realtime two-decision
benchmark pair matched the expected 4/6-second reported simulation times.
The previously downloaded runtime was reused after all seven pinned
executable/content files were rehashed; this was not a fresh OS installation.

The exported source passed 716 tests and 407 subtests, with one optional
serving test skipped. The full repository passed 736 tests and 407 subtests,
with the same skip. See the [environment, hashes and run results](../validation/realtime-server-20261010.json).
Windows, WSL2/Dozen and minimum GPU/VRAM requirements remain unvalidated.

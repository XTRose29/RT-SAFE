# Real-time release scope

This source distribution covers the original five-map benchmark only. It
excludes the website, map-transfer implementation, Unreal binaries, generated
campaigns and credentials. `SOURCE_MANIFEST.json` records each shipped file
and the originating development commit.

Fresh-install evidence and example comparisons are linked from
[examples](../examples/README.md). Runtime checks test rendering, Blueprint
availability, state APIs and pause/resume timing. They do not establish exact
historical model scores, native Windows/WSL2 support or minimum GPU memory.

Validated on 7 October 2026:

- The full upstream Linux ZIP passed its pinned SHA-256; its separately fetched
  real-time chunk was installed into the fresh extraction and matched exactly.
- A new Python environment passed 709 tests and 407 subtests; one optional
  serving test was skipped. A fresh Ubuntu CI runner also passed the exported
  source's dependency installation, tests and planning check.
- All five maps passed live runtime checks. Full benchmark scene generation
  and two policy decisions were also exercised on every map.
- Three static/real-time pairs matched the two-decision timing reference:
  4 seconds static, 6 seconds real-time, without model credentials.
- Greedy completed RT10 task 0 in both modes with 30 decisions and reported
  simulation times of 81 and 111 seconds.
- Codex CLI returned two valid GPT-6 Astra actions in each mode; the real-time
  trace confirmed concurrent world evolution during inference.

The GPU checks used fresh runtime, Python and cache directories on an existing
Ubuntu 24.04 host with an RTX A5000. The separate clean Ubuntu CI run did not
have a GPU. See [runtime evidence](../validation/realtime-fresh-runtime.json),
[repeat comparisons](../validation/realtime-fresh-examples.json), and
[benchmark scene evidence](../validation/realtime-fresh-campaign.json).

The original code license and editable Unreal project are not supplied by
this release. Third-party engine/content redistribution remains upstream.

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

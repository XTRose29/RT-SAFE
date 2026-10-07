# RT-SAFE real-time benchmark

The world does not pause while an agent thinks. This release contains the
original 36 routes across five SimWorld/Unreal maps, the Python evaluation
code, and tools to obtain and validate the matching public Linux runtime.

## Install

Use Linux x86-64 with Python 3.12 and an NVIDIA GPU/driver for live rendering.
From this directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt -c requirements.lock.txt
python -m pytest -q
python benchmark/run.py run smoke --plan-only
```

Then follow [runtime setup](docs/REALTIME_ADDON.md). The pinned 26 GB public
SimWorld archive already includes our real-time content. If you have the
matching base without that content, the separate fetch transfers about 42 MB.
No Unreal editor or model credentials are needed for runtime validation or
the [reproducible examples](examples/README.md).

See [installation](docs/INSTALL.md), [protocol](docs/PROTOCOL.md), and
[validation scope](docs/RELEASE_STATUS.md). Model credentials are needed only
when evaluating a hosted policy; the deterministic baselines require none.

## What this package contains

- Original maps RT10, RT12, RT15, RT18, RT20 and their authored routes.
- Timing, hazard attribution, code baselines, model adapters, and tests.
- Separate real-time download/installation tools and exact runtime hashes.
- Small examples with expected results and provenance.

This package excludes the website and the separate map-transfer project.
Runtime binaries and third-party Unreal assets are downloaded from SimWorld,
not mirrored here. The content add-on requires the exact compatible base;
it is not a standalone simulator. Native Windows and WSL2 are unvalidated.

## Results and license

The examples are installation checks, not a reproduction of the historical
paper leaderboard. Exact paper reproduction also requires the original model
versions, transport and campaign configuration. Keep new results labeled with
their actual settings. See [third-party notices](THIRD_PARTY_NOTICES.md).
The original RT-SAFE code license remains pending the authors' selection;
bundled SimWorld retains its Apache-2.0 license.

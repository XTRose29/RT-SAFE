# Separate real-time runtime add-on

The original Linux runtime stores the RT-SAFE additions in
`SimWorld/Content/Paks/pakchunk3002-Linux.pak`. This is a separate cooked content
chunk of about 36 MB. It can be installed independently of the much larger
SimWorld base. It is **not a standalone simulator**.

The chunk contains the five levels (`RT10`, `RT12`, `RT15`, `RT18`, `RT20`),
RT agent and hazard Blueprints, traffic controllers and pedestrian signals,
materials, textures, sports-ball assets, and two Vulkan shader archives. The
Python real-time execution and safety logic are already in this repository.

## Download the supported runtime

The specific legacy build is publicly available from
[SimWorld's official Base20260313 archive](https://huggingface.co/datasets/SimWorld-AI/SimWorld/tree/31bc15758b66cfd2effb93e2183a1d9ee76363c4/Base20260313).
Earlier RT-SAFE documentation overlooked this upstream package. It already
contains the RT-SAFE chunk, so a new installation does **not** need a second
add-on download. Download the pinned version, verify it, and extract it:

```bash
curl -L --fail --retry 3 -o Linux.zip \
  'https://huggingface.co/datasets/SimWorld-AI/SimWorld/resolve/31bc15758b66cfd2effb93e2183a1d9ee76363c4/Base20260313/Linux.zip?download=true'
echo '3a0ca8d98ba1de099541549826dc974007243ad4f5d04d0694960cbfe523287f  Linux.zip' | sha256sum -c -
python3.12 -m zipfile -e Linux.zip .
chmod +x Linux/SimWorld.sh Linux/SimWorld/Binaries/Linux/SimWorld
```

The ZIP is 25,950,358,766 bytes (about 26 GB); allow about 60 GB of free disk
space for the download, extracted runtime and Python environment. Extraction
uses Python's standard library, so a separate `unzip` program is not required.
Its checksum comes from upstream LFS metadata. This release
does not mirror the archive or change upstream asset terms.

The full archive was downloaded without authentication and verified against
that SHA-256. A fresh extraction, new Python environment, and separate cache
directories passed live checks on all five maps. The independently downloaded
real-time chunk was also installed into this fresh base and matched the ZIP's
copy byte for byte. See [fresh runtime validation](../validation/realtime-fresh-runtime.json),
[upstream verification](../validation/realtime-addon-upstream.json), and the
[base/add-on hashes](../validation/realtime-addon-manifest.json).

## Fetch the real-time part separately

If you already have the **matching** base but need its real-time chunk:

```bash
python tools/fetch_realtime_addon.py --output downloads/pakchunk3002-Linux.pak
python tools/runtime_addon.py install-pak \
  --runtime /path/to/Linux \
  --pak downloads/pakchunk3002-Linux.pak
```

The standard-library downloader reads only the relevant ZIP ranges from
SimWorld's own hosting (about 42 MB transferred in validation), then verifies
the 36,891,601-byte chunk against its pinned SHA-256. The installer verifies
the executable and required content hashes before installing and refuses to overwrite a different
chunk. Neither script requires model credentials, an Unreal editor, or a
Hugging Face token. Do not substitute a newer generic SimWorld build.

## Compatibility

The add-on targets the original **Linux x86-64, Unreal Engine 5.3.2** build
(`5.3.2-29314046+++UE5+Release-5.3`). It does not target the newer NYC/SPEAR
project. Matching the Unreal version alone is insufficient: the installer
checks the executable and base content against the manifest's SHA-256 hashes.
UnrealCV is compiled into the base executable, not provided by this content
chunk. The add-on therefore cannot add time-control RPCs to an arbitrary engine.

There is no validated native Windows package, WSL2 configuration, minimum GPU
specification, or editable Unreal source project in this add-on release.
Linux cooked assets must not be advertised as a Windows build.

## Prepare a self-contained add-on archive (maintainers)

Given an authorized copy of the original compatible runtime:

```bash
python tools/runtime_addon.py build \
  --runtime /path/to/original/Linux-runtime \
  --archive release-artifacts/rt-safe-realtime-linux-candidate.tar \
  --manifest release-artifacts/realtime-addon-manifest.json
sha256sum release-artifacts/rt-safe-realtime-linux-candidate.tar
```

This copies only the existing RT chunk, a manifest, and the standard-library
installer into the archive. It does not repack the base engine, model weights,
logs, or credentials. The manifest identifies the exact prerequisite files.
Hashing the base reads roughly 25 GB and can take several minutes. The builder
refuses to replace an existing archive; use a new filename for a new candidate.
The chunk also includes sports-ball assets and textures. Obtain their notices
and redistribution terms before mirroring the binary under RT-SAFE; the public
fetcher above instead retrieves it directly from SimWorld's existing hosting.

## Install on a matching base

For a maintainer-prepared archive, verify its published checksum. Stop the selected
runtime, then use a separate writable copy; do not install into an active
shared experiment runtime.

```bash
python tools/runtime_addon.py check \
  --runtime /path/to/Linux-runtime \
  --archive /path/to/rt-safe-realtime-linux-candidate.tar
python tools/runtime_addon.py install \
  --runtime /path/to/Linux-runtime \
  --archive /path/to/rt-safe-realtime-linux-candidate.tar
```

Both commands verify all prerequisite fingerprints in the manifest and the payload checksum.
`check` leaves no installed add-on. `install` writes just the missing RT chunk;
reinstalling identical bytes is harmless, and a different installed chunk is
never overwritten. The `Paks` directory must be an actual writable directory,
not a link into another installation. The archive also contains `install.py`
for users without the repository; it accepts the same commands.

## Validate in Unreal before evaluating models

After installing the Python requirements from [INSTALL.md](INSTALL.md), run:

```bash
python tools/check_runtime_addon.py \
  --launcher /path/to/Linux-runtime/SimWorld.sh \
  --gpu 0 --port 19091 --output results/runtime-addon-smoke
```

The checker launches a fresh engine per level, uses an otherwise unused port,
and stops only its own process. It does not call model APIs. It checks:

- Required agent, hazard, traffic, and base Blueprint classes can be spawned.
- Agent safety and pedestrian/vehicle signal state APIs return their fields.
- An actor with a queued move stays still during a paused wall-clock interval
  and moves after the world resumes.
- The selected time-advance API advances an interval and returns to a paused world.
- The engine can return a nonblank rendered camera frame.

The output directory must be new. It contains per-map engine logs, images, and
`report.json`. A failed check exits nonzero; a listening UnrealCV port alone is
not treated as proof of compatibility. `--levels RT10` provides a shorter first
check. These checks establish basic runtime behavior, not historical score
reproduction, all hazard outcomes, or hardware minimums.

All five maps passed these live checks from the freshly downloaded runtime on
Linux with an NVIDIA RTX A5000. See [the validation record](../validation/realtime-fresh-runtime.json).
The standalone source passed 709 Python tests and 407 subtests, with one
optional serving test skipped, in a clean dependency installation.

For a manual benchmark launch:

```bash
/path/to/Linux-runtime/SimWorld.sh /Game/RealTimeBench/Maps/RT10 \
  -RenderOffScreen -windowed -graphicsadapter=0 -cvport 19091 \
  -ResX=720 -ResY=640 -FPSMAX=30 -noraytracing
```

Wait for `Engine is initialized. Leaving FEngineLoop::Init()` in the console
before connecting a runner. The UnrealCV listener opens earlier than this.

Before a benchmark run, select the validated resume/pause time-advance API:

```bash
export SIMWORLD_UE_LAUNCHER=/path/to/Linux-runtime/SimWorld.sh
export SIMWORLD_TIME_ADVANCE_MODE=resume_pause
export SIMWORLD_OBSERVATION_WIDTH=720
export SIMWORLD_OBSERVATION_HEIGHT=640
export SIMWORLD_SKIP_INITIAL_SETRES=1
export SIMWORLD_SKIP_ASYNC_SKINNED_ASSET_COMPILATION=1
```

Then configure a model and follow [INSTALL.md](INSTALL.md). The
`benchmark/run.py doctor smoke` command checks the Qwen suite and also expects
a Qwen model or endpoint; it is not required for model-free or hosted CLI
examples. Keep the base
manifest, add-on checksum, code commit and resolved suite configuration with
the results. The pinned full download above provides the matching base for
new users; the smaller separate fetch is for existing matching installations.

## Model-free benchmark smoke

The real benchmark runner was exercised in three independent pairs of two
greedy decisions on RT10 task 0 in static and real-time mode, with one second
of thinking per decision. All six runs completed, recorded zero model tokens,
and detected an oil hazard. Reported simulation time was 4 seconds for static
and 6 seconds for real-time, matching the two added thinking intervals. These
are short smoke runs, not successful full-route completions or paper results.
See [the fresh repeated comparisons](../validation/realtime-fresh-examples.json).

To repeat one condition, launch RT10 using the manual command above and run
this from the repository root (restart Unreal before the other condition):

```bash
export PYTHONPATH="$PWD/SimWorld:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export SIMWORLD_TIME_ADVANCE_MODE=resume_pause
export SIMWORLD_OBSERVATION_WIDTH=720
export SIMWORLD_OBSERVATION_HEIGHT=640
export SIMWORLD_SKIP_INITIAL_SETRES=1
export SIMWORLD_SKIP_ASYNC_SKINNED_ASSET_COMPILATION=1
python evaluation/run_code_baselines.py \
  --task-index 0 --baselines greedy --env-modes realtime --time-modes sync \
  --max-steps 2 --internal-latency-seconds 1 --max-rounds 1 \
  --ue-port 19091 --suite-name addon-smoke-realtime --no-record-per-step
```

For the static check, change `--env-modes static` and use a new suite name.
The runner exits nonzero after a condition error or missing result. A valid
smoke result must have `status: completed` and a result with two decisions.
The explicit 720×640
observation settings match the normal benchmark runner and keep all seven
policy waypoint markers visible.

For automatic engine launch, cleanup and comparison of both modes, use
the [one-command examples](../examples/README.md).

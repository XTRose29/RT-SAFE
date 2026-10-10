# Run on a native Linux GPU server

This path runs the original RT-SAFE maps and the actual benchmark runner
without a model server or API key. Use it before configuring a model.
It targets native Linux with a working Vulkan-capable NVIDIA driver;
Windows, WSL2 and minimum GPU/VRAM requirements are not validated.

## 1. Install the published Python source

On Ubuntu 24.04, install system prerequisites, then clone the repository:

```bash
sudo apt-get update
sudo apt-get install -y python3.12-venv git libgl1 libegl1 libglib2.0-0t64 ffmpeg curl
git clone https://github.com/XTRose29/RT-SAFE.git
cd RT-SAFE
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt -c requirements.lock.txt
python -m pip check
```

Install the NVIDIA driver through the host's normal driver setup. Confirm
`nvidia-smi` works and choose an available GPU. CUDA model inference working
does not alone prove that Unreal can use the GPU for Vulkan rendering.

## 2. Obtain the matching Unreal runtime

Follow the [pinned download and SHA-256 verification instructions](REALTIME_ADDON.md#download-the-supported-runtime).
The roughly 26 GB archive already includes RT10/RT12/RT15/RT18/RT20 and the
required assets. Allow about 60 GB for download, extraction and dependencies.
No Unreal editor or source project is needed to run this packaged Linux build.

Set the launcher to your extracted copy, and select an available GPU and port:

```bash
export SIMWORLD_UE_LAUNCHER=/absolute/path/to/Linux/SimWorld.sh
export RTSAFE_GPU=0
export RTSAFE_PORT=19091
chmod +x "$SIMWORLD_UE_LAUNCHER" "$(dirname "$SIMWORLD_UE_LAUNCHER")/SimWorld/Binaries/Linux/SimWorld"
```

Each command below launches and cleans up its own Unreal process. Do not
start a separate engine on the same port. Run commands from the repository
root and use new output directories for each attempt.

## 3. Check all five maps

```bash
python tools/check_runtime_addon.py \
  --launcher "$SIMWORLD_UE_LAUNCHER" \
  --gpu "$RTSAFE_GPU" --port "$RTSAFE_PORT" \
  --output results/first-runtime-check
```

Successful validation exits with code 0 and writes `report.json` with
`passed: true`, five map records, screenshots, and engine logs. To diagnose a
first installation faster, add `--levels RT10` before checking every map.
See [software rendering diagnostics](REALTIME_ADDON.md#software-rendering-and-wsl2)
if an engine connects but movement fails.

## 4. Run an actual benchmark pair

```bash
python tools/run_realtime_examples.py \
  --launcher "$SIMWORLD_UE_LAUNCHER" \
  --gpu "$RTSAFE_GPU" --port "$RTSAFE_PORT" \
  --output results/first-benchmark-pair
```

This launches independent RT10 engines for static and real-time conditions
and runs the deterministic greedy policy. A passing `report.json` contains
two completed runs, two decisions per run, zero model tokens, and reported
simulation time of 4 seconds static / 6 seconds real-time. The two-decision
limit intentionally stops before the destination, so `success: false` is
expected. Reported simulation time is runner accounting, not an independent
measurement of every physics tick. See [the example contract](../examples/README.md).

For a route-length run, follow the [complete greedy route example](../examples/full-route/README.md).
For model evaluation, continue with [model configuration](INSTALL.md#3-configure-unreal-and-a-model).
`doctor smoke` checks a Qwen endpoint as well as the runtime; it is not a
prerequisite for either model-free command above.

## Validation evidence and limits

See the [2026-10-10 server validation record](../validation/realtime-server-20261010.json)
for the environment, runtime hashes, five-map checks and benchmark pair.
This run used a new Python environment and the standalone source export,
with a previously downloaded runtime whose pinned files were rehashed.
These checks establish installation and basic runtime/benchmark behavior.
They do not reproduce the paper's model scores or validate other operating
systems, all hardware, or every hazard outcome.

When reporting a failure, include the code revision, command, OS/GPU/driver,
`report.json`, and the engine/console/runner logs from the output directory.
No model credentials are needed for these checks.

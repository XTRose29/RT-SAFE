# Installation and first run

## 1. Install the Python environment

Use Linux x86-64 and Python 3.12. On minimal Ubuntu 24.04, install the system
libraries and tools first:

```bash
sudo apt-get update
sudo apt-get install -y python3.12-venv libgl1 libegl1 ffmpeg curl
```

From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt -c requirements.lock.txt
export PYTHONPATH="$PWD/SimWorld:$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

Install FFmpeg for video export. A graphical workstation or suitable offscreen configuration and NVIDIA driver are needed for live Unreal rendering. Local Qwen inference needs its own GPU serving environment (a CUDA-compatible PyTorch build plus `requirements-serving.txt`); hosted API evaluation does not require a local model GPU. Hardware needs depend on the runtime and selected checkpoint.

Install the NVIDIA driver through your system's normal driver setup. A Python
virtual environment does not install a GPU driver. The tested dependency
snapshot targets Linux x86-64 / Python 3.12.

## 2. Explore without Unreal or API credentials

```bash
python benchmark/run.py list-configs
python benchmark/run.py run smoke --plan-only
python benchmark/experiments.py plan --program-name qwen3vl_open_source_v1
python -m pytest -q
```

The model-server tests are skipped when the optional PyTorch/Transformers serving stack is absent. Planning prints the resolved commands. It does not call models or start the simulator.

## 3. Configure Unreal and a model

Follow [ASSETS.md](ASSETS.md) to obtain the compatible runtime.

For the original Linux 5.3.2 runtime, use the pinned SimWorld download in
[REALTIME_ADDON.md](REALTIME_ADDON.md), then run its model-free engine check.
That guide also supports downloading the real-time content separately when
you already have the matching base.

Set:

```bash
export SIMWORLD_UE_LAUNCHER=/absolute/path/to/SimWorld.sh
export SIMWORLD_TIME_ADVANCE_MODE=resume_pause
export SIMWORLD_OBSERVATION_WIDTH=720
export SIMWORLD_OBSERVATION_HEIGHT=640
export SIMWORLD_SKIP_INITIAL_SETRES=1
export SIMWORLD_SKIP_ASYNC_SKINNED_ASSET_COMPILATION=1
```

For model-free validation, use the [one-command examples](../examples/README.md).
The following commands configure the Qwen suite; hosted API/CLI users can
continue to section 4 without a local Qwen model or server.

```bash
export SIMWORLD_QWEN_MODEL=/absolute/path/to/Qwen3-VL-8B-Instruct
python benchmark/run.py doctor smoke
```

The suite runner supports an existing OpenAI-compatible model endpoint and managed local inference. Its default external endpoint is `http://127.0.0.1:30001/v1`. Use `--runner-arg` to override individual runner settings; consult `python evaluation/run_qwen3vl8b_all_maps.py --help` for the complete interface.

```bash
python benchmark/run.py run smoke --suite-name first-smoke \
  --runner-arg=--qwen-mode --runner-arg=external
python benchmark/run.py run all_tasks_easy_realtime_collision \
  --suite-name first-easy-realtime --runner-arg=--qwen-mode --runner-arg=external
```

Use `--resume` with the same suite name to continue. Outputs are stored under `results/`, which is excluded from git. The default suite is easy/realtime, 36 routes. It does not by itself reproduce the manuscript's hard/static comparison.

## 4. Hosted models

The single-task API entry point expects a running UE server. Use the manual
RT10 launch command in [REALTIME_ADDON.md](REALTIME_ADDON.md), keep the common
environment settings above, and wait for engine initialization. Preview a task:

```bash
python evaluation/run_openai_benchmark.py \
  --model gpt-6-astra --reasoning-effort medium \
  --task-file data/map1_10roads/tasks.json --task-index 0 --ue-port 19091 --dry-run
```

To execute, set `OPENAI_API_KEY` in your shell and omit `--dry-run`. Use the explicit provider/API-mode options for OpenRouter or authenticated CLI transports. Never commit credentials or generated account-usage manifests. Provider availability and pinned model IDs can change; record the resolved serving identity with every result.

## 5. Project website

See [WEBSITE.md](WEBSITE.md). The website can be built and used without Python, Unreal, or model access.

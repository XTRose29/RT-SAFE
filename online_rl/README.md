# Experimental online RL with VAGEN

> Restored from research commit `1bbb87de`. Adapter and fleet unit tests are included; end-to-end training has not been validated in this release. This pipeline is separate from the paper's offline BC/RL experiment. External training repositories and runtime assets are required.


This directory preserves an experimental run kit for training the
SimWorld-RealTime benchmark with the VAGEN/verl pipeline introduced by
simworld-nav PR #3. The adapter uses the benchmark's real prompts, ordered
camera images, parser, decision latency, physics, hazards and terminal rules.
Privileged simulator state is used only for reward and telemetry; it is never
included in the policy observation.

## Included components

- `vagen_env.py`: async VAGEN `GymImageEnv` adapter with exclusive UE leasing.
- `realtime_backend.py`: one external model action at a time through the
  maintained `WorldManager` and `RTAgent` implementation.
- `train_vagen.yaml`: 1,000 training seeds, RT10 easy tasks 0-7, 24 turns.
- `val_vagen.yaml`: 32 fixed held-out seeds, disjoint from training.
- `start_fleet.sh`: launches and owns one packaged UE process per rollout GPU.
- `smoke_env.py`: a real reset, image observation, parsed action and reward.
- `preflight.py`: files, dataset, endpoint, VAGEN, patch, model and CUDA checks.
- `train_grpo_vagen.sh`: patched multimodal GRPO launcher with auditable dumps.

## 1. Install the pinned training environment

Use Python 3.12. These are the versions exercised with simworld-nav PR #3:

```bash
conda create -n simworld-rt-rl python=3.12 -y
conda activate simworld-rt-rl
pip install torch==2.8.0
pip install vllm==0.11.0 ray==2.53.0 transformers==4.57.1 accelerate==1.12.0
pip install cairosvg pillow numpy pyyaml gymnasium pydantic jsonschema networkx pytest
```

Fetch PR #3 and the pinned VAGEN/verl revisions:

```bash
git clone https://github.com/mk322/simworld-nav.git /path/to/simworld-nav-pr3
cd /path/to/simworld-nav-pr3
git fetch origin pull/3/head:online-rl-pr3
git checkout online-rl-pr3
test "$(git rev-parse HEAD)" = 0c6798f62280cc6fd6c8f9f999064e5e90e91821

mkdir -p vendor
git clone https://github.com/ymzhang0303/VAGEN.git vendor/vagen
git -C vendor/vagen checkout b93deaa
git clone https://github.com/JamesKrW/verl.git vendor/vagen/verl
git -C vendor/vagen/verl checkout 3fe0a29
pip install -e vendor/vagen -e vendor/vagen/verl
```

Install the maintained benchmark's local UnrealCV/runtime dependencies too:

```bash
cd /path/to/SimWorld-RealTime
python -m pip install -r requirements.txt
```

Copy `online_rl/env.example` to a private shell file and replace every path.
The packaged Unreal build and Qwen3-VL-4B-Instruct checkpoint are external
assets and must already exist on the machine.

```bash
cd /path/to/SimWorld-RealTime
cp online_rl/env.example /tmp/simworld-rt-online-rl.env
$EDITOR /tmp/simworld-rt-online-rl.env
source /tmp/simworld-rt-online-rl.env
python3 online_rl/preflight.py --phase setup
```

## 2. Start the UE rollout fleet (terminal 1)

Reserve separate physical GPU sets for UE and training. The example uses GPUs
0-3 for four UE instances and GPUs 4-7 for four training workers. Do not run
this alongside a benchmark campaign that is already using those UE ports or
GPUs.

```bash
cd "$REALTIME_ROOT"
source /tmp/simworld-rt-online-rl.env
bash online_rl/start_fleet.sh
```

Wait for `UE fleet ready`. Leave the terminal open; Ctrl-C shuts down exactly
the processes it launched and marks the endpoint manifest stopped.

## 3. Prove the complete environment path (terminal 2)

The smoke test leases one fleet member, loads RT10, obtains the official visual
observation, submits `wait(1)`, and prints the reward decomposition and state.

```bash
cd "$REALTIME_ROOT"
source /tmp/simworld-rt-online-rl.env
python3 online_rl/preflight.py --phase runtime
python3 online_rl/smoke_env.py
```

Success is a JSON object with `"ok": true`, at least one image, an endpoint
ID, `turns`/state information, and finite wall times. The smoke episode is
deliberately capped at one action, so `done: true` and `termination:
"max_turns"` are expected. A failed smoke means do not start Ray/VAGEN yet.

## 4. Run one no-learning GRPO step (terminal 2)

This checks batching, 16 grouped rollouts, log-probabilities, reward transport,
the actor update path and checkpoint writing while keeping weights unchanged.
Validation is disabled for this plumbing run because its 32 real episodes add
roughly an hour and do not make the smoke more informative.

```bash
cd "$REALTIME_ROOT"
source /tmp/simworld-rt-online-rl.env
export ACTOR_LR=0
export TOTAL_STEPS=1
export VAL_BEFORE_TRAIN=False
export TEST_FREQ=0
export EXPERIMENT_DIR="$REALTIME_ROOT/exps/grpo_realtime_lr0"
bash online_rl/train_grpo_vagen.sh
```

Inspect `rollouts/`, `train.log`, and the reward decomposition printed by the
adapter smoke. Within every four-trajectory GRPO group, actions and scalar
returns should not all be identical, and the run should have no missing-image,
parse-error storm, endpoint timeout, NaN or zero-length response. Learning rate
zero does not mean `grad_norm` must be zero; it means the optimizer must not
change parameters.

## 5. Start training

Open a clean terminal or unset the smoke overrides. Ten steps is the sensible
first learning experiment; only scale to 40 after reviewing paired held-out
episodes and reward components.

```bash
cd "$REALTIME_ROOT"
source /tmp/simworld-rt-online-rl.env
export ACTOR_LR=1e-6
export TOTAL_STEPS=10
export VAL_BEFORE_TRAIN=True
export TEST_FREQ=10
export EXPERIMENT_DIR="$REALTIME_ROOT/exps/grpo_realtime_10step"
bash online_rl/train_grpo_vagen.sh
```

Important defaults are `TRAIN_BS=4`, `GROUP=4` (16 real episodes per update),
`AGENT_WORKERS=4`, four training GPUs, a 24-action episode cap, KL loss 0.005,
FSDP2 with CPU offload, and one held-out validation at the start and end of a
10-step run. `SIMWORLD_RL_UE_ENDPOINTS` is shared safely: file locks ensure
only one Ray worker drives a stateful UE endpoint at a time.

## Timing and capacity planning

The measured maintained-benchmark artifacts average 24.2 seconds per decision
(median 21.5), of which 4.6 seconds is model inference. A 24-decision capped
episode is therefore about 8.6-9.7 minutes serially. The useful formula is:

```text
update wall ~= ceil((TRAIN_BS * GROUP) / UE instances)
              * decisions/episode * seconds/decision
              + optimizer time
```

With four UE instances and four training GPUs, plan on about 45-65 minutes per
16-episode update after warm-up: roughly 35-40 minutes of rollout waves plus
5-20 minutes for a 4B FSDP update and operational overhead. The first
step can be slower due to model and Ray startup. A 32-episode validation costs
about 60-80 minutes on four UEs. This gives these practical budgets:

| Run | Approximate wall time |
|---|---:|
| adapter smoke (one action) | 2-6 min including world reset/cleanup |
| one LR=0 GRPO step, no validation | 45-75 min plus first startup |
| 10 updates + initial/final validation | 10-14 h |
| 40 updates with validation every 10 | 32-48 h |

Six UE instances can reduce a 16-episode rollout to roughly 23-26 minutes and
held-out validation to 35-55 minutes, but only if those extra UE GPUs are truly
available. A full, uncapped benchmark episode is much longer: the observed
median was 70 decisions and 27.3 minutes, so use the 24-action curriculum for
the first online-RL runs.

For comparison, simworld-nav's symbolic courier environment has already run at
about 6.4 minutes total for 12 steps on a 2B LoRA and 18 minutes for 12 steps on
a 7B configuration; its live courier fleet sustained 92.5 episodes/hour. Those
numbers should not be used to budget SimWorld-RealTime: Unreal camera/action
physics and real decision latency make the RealTime pipeline roughly an
overnight 10-step experiment, not a minutes-long job.

## Stop and troubleshoot

- Stop training with Ctrl-C, then stop terminal 1 with Ctrl-C. The fleet
  launcher performs process-group cleanup and rewrites the manifest as stopped.
- `cannot connect`: inspect `$UE_LOG_DIR`, port ownership and packaged-build map
  availability before retrying.
- vLLM free-memory error: UE and trainer GPU sets overlap or another job took a
  training card.
- NCCL P2P and SHM are disabled by default for the tested A5000 topology;
  override those variables only after validating the machine's topology.
- Prompts mention images but rollouts contain none: verify `cairosvg` and Pillow
  in the training environment, then rerun `smoke_env.py`.
- All rewards equal inside every group: do not continue training. Review
  `reward_terms`, parsed actions, task/seed grouping and whether the model is
  producing valid `Action`/`Param` responses.

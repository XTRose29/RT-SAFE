#!/usr/bin/env bash
# VAGEN/verl GRPO launcher for online_rl.vagen_env.SimWorldRealTimeGymEnv.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
: "${VAGEN_ROOT:?set VAGEN_ROOT to the VAGEN checkout}"
: "${SIMWORLD_NAV_ROOT:?set SIMWORLD_NAV_ROOT to the simworld-nav PR #3 checkout}"
: "${MODEL_PATH:?set MODEL_PATH to a local Hugging Face checkpoint}"
: "${SIMWORLD_RL_UE_ENDPOINTS:?set SIMWORLD_RL_UE_ENDPOINTS to the ready fleet manifest}"
: "${N_GPUS:?set N_GPUS to the number of CUDA_VISIBLE_DEVICES reserved for training}"

TRAIN_DATA=${TRAIN_DATA:-${REPO}/online_rl/train_vagen.yaml}
VAL_DATA=${VAL_DATA:-${REPO}/online_rl/val_vagen.yaml}
EXPERIMENT_DIR=${EXPERIMENT_DIR:-${REPO}/exps/grpo_realtime}
mkdir -p "${EXPERIMENT_DIR}"
export PYTHONPATH="${REPO}:${VAGEN_ROOT}:${PYTHONPATH:-}"
export WANDB_MODE=${WANDB_MODE:-offline}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-1}
EXTRA_TRAINER_ARGS=()
if [[ -n "${TOTAL_STEPS:-}" ]]; then
  EXTRA_TRAINER_ARGS+=("trainer.total_training_steps=${TOTAL_STEPS}")
fi

# The simworld-nav PR carries fixes for multimodal truncation, generation
# budgets and rollout-correction fields that upstream VAGEN may not contain.
# Point this at a checkout of PR #3 to verify/apply those idempotent patches.
export PYTHONPATH="${SIMWORLD_NAV_ROOT}:${PYTHONPATH}"
"${PYTHON_BIN}" "${REPO}/online_rl/preflight.py" --phase train

for patch in agent_loop_qwen3vl_rope \
             agent_loop_image_safe_truncation \
             agent_loop_generation_budget \
             policy_loss_rollout_correction_field \
             multiturn_image_safe_truncation \
             silent_zero_scores; do
  "${PYTHON_BIN}" -m "embodiedbench.training.vagen.patches.${patch}"
done

cd "${VAGEN_ROOT}"
"${PYTHON_BIN}" -m vagen.main_ppo \
  --config-path="${VAGEN_ROOT}/vagen/configs" \
  --config-name=vagen_multiturn \
  +env_registry.RealTime=online_rl.vagen_env.SimWorldRealTimeGymEnv \
  data.train_files="${TRAIN_DATA}" \
  data.val_files="${VAL_DATA}" \
  data.train_batch_size=${TRAIN_BS:-4} \
  data.max_prompt_length=${PROMPT_LEN:-4096} \
  data.max_response_length=${RESP_LEN:-16384} \
  algorithm.adv_estimator=grpo \
  algorithm.kl_ctrl.kl_coef=0.0 \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=${ACTOR_LR:-1e-6} \
  actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_BS:-4} \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=${USE_KL:-True} \
  actor_rollout_ref.actor.kl_loss_coef=${KL_COEF:-0.005} \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.offload_policy=True \
  actor_rollout_ref.ref.strategy=fsdp2 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.name=${ROLLOUT_BACKEND:-vllm} \
  actor_rollout_ref.rollout.mode=async \
  ++actor_rollout_ref.rollout.agent.num_workers=${AGENT_WORKERS:-4} \
  actor_rollout_ref.rollout.n=${GROUP:-4} \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_MEM:-0.4} \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_SEQS:-32} \
  actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_TOKENS:-32768} \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${VAGEN_ROOT}/vagen/configs/agent.yaml" \
  trainer.logger="['console']" \
  trainer.n_gpus_per_node=${N_GPUS:-1} \
  trainer.nnodes=1 \
  trainer.save_freq=${SAVE_FREQ:-10} \
  trainer.test_freq=${TEST_FREQ:-10} \
  trainer.val_before_train=${VAL_BEFORE_TRAIN:-True} \
  trainer.total_epochs=${EPOCHS:-10} \
  trainer.project_name=verl_vagen \
  trainer.experiment_name=grpo_simworld_realtime \
  trainer.default_local_dir="${EXPERIMENT_DIR}/checkpoints" \
  trainer.rollout_data_dir="${EXPERIMENT_DIR}/rollouts" \
  trainer.validation_data_dir="${EXPERIMENT_DIR}/validation" \
  "${EXTRA_TRAINER_ARGS[@]}" \
  2>&1 | tee "${EXPERIMENT_DIR}/train.log"

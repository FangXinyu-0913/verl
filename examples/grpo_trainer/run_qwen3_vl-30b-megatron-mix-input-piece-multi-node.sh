set -eux
umask 000
ENGINE=${1:-vllm}

export NCCL_TIMEOUT=1800
export TORCH_DISTRIBUTED_TIMEOUT=1800
export TORCH_NCCL_BLOCKING_WAIT=1
export CUDA_DEVICE_MAX_CONNECTIONS=1 # For megatron communication/computation overlapping

export VLLM_ALLREDUCE_USE_SYMM_MEM=0 # for vllm0.11.0 with TP
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

VERL_ROOT_DIR=/mnt/shared-storage-user/fangxinyu/jigsaw_project/RealJigsaw-RL/verl

cd $VERL_ROOT_DIR
ray stop --force

export RAY_MASTER_PORT=6379
export RAY_DASHBOARD_PORT=8265
export NODE_RANK=${NODE_RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"} # Default to localhost if not set
export RAY_ADDRESS="http://$MASTER_ADDR:$RAY_DASHBOARD_PORT"



GEN_TP=${GEN_TP:-8}
CP=${CP:-2}
TP=${TP:-2}
PP=${PP:-1}
EP=${EP:-8}
ETP=${ETP:-1}

train_path='/mnt/shared-storage-user/mllm/fangxinyu/jigsaw/train_data_for_verl/train_3KindsMix_pieces_resolution448_OneStage.parquet'
test_path='/mnt/shared-storage-user/mllm/fangxinyu/jigsaw/train_data_for_verl/train_3KindsMix_pieces_resolution448_OneStage.parquet'

HF_MODEL_PATH="/mnt/shared-storage-user/large-model-center-share-weights/hf_hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/4b184fbdab8886057d8d80c09f35bcfc65fe640e"

PROJECT_NAME="verl_grpo_jigsaw_Mix3Kinds_input_piece_OneStage"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo "[INFO] TIMESTAMP: $TIMESTAMP"
export EXPERIMENT_NAME="qwen3_vl_30b_megatron_single_turn_without_shape"


if [ "$NODE_RANK" -eq 0 ]; then
    ###################################
    # HEAD NODE LOGIC (NODE_RANK == 0)
    ###################################
    echo "[INFO] This is the HEAD node (Rank 0) with Master Address: $MASTER_ADDR"

    # Start Ray Head
    ray stop -f
    ray start --head --node-ip-address="$MASTER_ADDR" --port="$RAY_MASTER_PORT" --dashboard-host=0.0.0.0 --dashboard-port="$RAY_DASHBOARD_PORT" --num-gpus=8

    # Wait for all worker nodes to connect
    sleep 20

    LOG_FILE="logs/$PROJECT_NAME/$EXPERIMENT_NAME/log.txt"
    mkdir -p "$(dirname "$LOG_FILE")"
    ray job submit --address="$MASTER_ADDR:$RAY_MASTER_PORT" \
        -- python3 -m verl.trainer.main_ppo --config-path=config \
        --config-name='ppo_megatron_trainer.yaml'\
        algorithm.adv_estimator=grpo \
        data.train_files="$train_path" \
        data.val_files="$test_path" \
        data.train_batch_size=64 \
        data.max_prompt_length=9192 \
        data.max_response_length=9192 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        actor_rollout_ref.model.path=$HF_MODEL_PATH \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size=64 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$PP \
        actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TP \
        actor_rollout_ref.actor.megatron.context_parallel_size=$CP \
        actor_rollout_ref.actor.megatron.expert_model_parallel_size=$EP \
        actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=$ETP \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.01 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=$GEN_TP \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=9192 \
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=9192 \
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=9192 \
        actor_rollout_ref.rollout.name=$ENGINE \
        +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
        actor_rollout_ref.rollout.n=8 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.actor.megatron.use_mbridge=True \
        actor_rollout_ref.actor.megatron.param_offload=False \
        actor_rollout_ref.actor.megatron.optimizer_offload=False \
        actor_rollout_ref.actor.megatron.grad_offload=False \
        actor_rollout_ref.ref.megatron.param_offload=False \
        +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1 \
        +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
        +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
        +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
        +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32 \
        +actor_rollout_ref.actor.megatron.override_transformer_config.moe_enable_deepep=True \
        +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=flex \
        +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
        +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
        +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
        +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True \
        +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True \
        +actor_rollout_ref.actor.megatron.override_transformer_config.moe_aux_loss_coeff=0.01 \
        +actor_rollout_ref.actor.megatron.override_transformer_config.moe_z_loss_coeff=0.001 \
        algorithm.use_kl_in_reward=False \
        trainer.critic_warmup=0 \
        trainer.logger='["console","tensorboard"]' \
        trainer.project_name=$PROJECT_NAME \
        trainer.experiment_name=$EXPERIMENT_NAME \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=2 \
        trainer.save_freq=50 \
        trainer.test_freq=10 \
        trainer.total_epochs=8 $@ \
        2>&1 | tee >(sed -r "s/\x1B\[[0-9;]*[mK]//g" > "$LOG_FILE")
    
    echo "[HEAD] Job finished."

else
    echo "[INFO] This is a WORKER node (Rank $NODE_RANK)."
    ray start --address="${MASTER_ADDR}:6379" --num-gpus=8

    echo "[WORKER] Started and connected to head node: ${MASTER_ADDR}:6379"
    echo "[WORKER] Monitoring head node status..."

    while true; do
        # if head node is not reachable, stop worker
        if ! ray status --address="${MASTER_ADDR}:6379" > /dev/null 2>&1; then
            echo "[WORKER] Head node unreachable. Stopping worker..."
            ray stop -f
            exit 0
        fi
        sleep 10  # check every 60 seconds
    done
fi


    # # Use aux_loss and z_loss to mitigate expert load imbalance when training MoE models
    # +actor_rollout_ref.actor.megatron.override_transformer_config.moe_aux_loss_coeff=0.01 \
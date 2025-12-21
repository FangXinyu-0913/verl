# run on 8xH100
# make sure your current working directory is the root of the project
# this is a verification training script, the parallel setting should be tuned to your model

set -x

export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0
export RUST_BACKTRACE=1
export HYDRA_FULL_ERROR=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

ulimit -n 65535

PROJECT_DIR="/mnt/shared-storage-user/fangxinyu/jigsaw_project/RealJigsaw-RL/verl"
CONFIG_PATH="$PROJECT_DIR/examples/jigsaw_trainer/config"


python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='jigsaw_multiturn_megatron_grpo_w_tool' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=32 \
    data.max_prompt_length=55152 \
    data.max_response_length=9192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=/mnt/shared-storage-user/large-model-center-share-weights/hf_hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/cadac78306af287f801b75a5565ede58f323f472 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=2 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=2 \
    actor_rollout_ref.actor.megatron.context_parallel_size=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.megatron.seed=42 \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=2 \
    actor_rollout_ref.ref.megatron.virtual_pipeline_model_parallel_size=2 \
    actor_rollout_ref.ref.megatron.context_parallel_size=2 \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.response_length=65536 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.megatron.param_offload=False \
    actor_rollout_ref.actor.megatron.optimizer_offload=False \
    actor_rollout_ref.actor.megatron.grad_offload=False \
    actor_rollout_ref.ref.megatron.param_offload=True \
    +actor_rollout_ref.ref.megatron.model_parallel_size=1 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=False \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=False \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=False \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name='verl_grpo_jigsaw_example500' \
    trainer.experiment_name='qwen3_vl_8b_megatron_w_edge_classification_tool_AND_resemble_tool_four_pieces' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.val_before_train=False \
    trainer.test_freq=10 \
    data.train_files=/mnt/shared-storage-user/fangxinyu/jigsaw_project/RealJigsaw-RL/get_data/train_data/train_four_pieces_resolution448_w_resemble_and_edge_classification_tool.parquet \
    data.val_files=/mnt/shared-storage-user/fangxinyu/jigsaw_project/RealJigsaw-RL/get_data/train_data/test_four_pieces_resolution448_w_resemble_and_edge_classification_tool.parquet \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="/mnt/shared-storage-user/fangxinyu/jigsaw_project/RealJigsaw-RL/verl/examples/sglang_multiturn/config/tool_config/jigsaw_restoration_tool_AND_edge_classification_config.yaml" \
    trainer.total_epochs=15 $@

#     actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=2 \


#     actor_rollout_ref.actor.use_dynamic_bsz=True \
# actor_rollout_ref.actor.ppo_max_token_len_per_gpu=9192 \
# actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
# actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=9192 \
# actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
# actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=9192 \

# actor_rollout_ref.ref.megatron.param_offload=True
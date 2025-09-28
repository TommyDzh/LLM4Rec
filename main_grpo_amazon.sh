set -x
MODEL_PATH=Qwen3-4B
NUM_GPUS=$(nvidia-smi -L | wc -l)
BATCH_SIZE=32
DATA_NAME=max@10-long@False
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=data/Movies/verl/$DATA_NAME/train.parquet \
    data.val_files=data/Movies/verl/$DATA_NAME/val.parquet \
    custom_reward_function.path=verl/utils/reward_score/regression.py \
    data.train_batch_size=$(($BATCH_SIZE * $NUM_GPUS)) \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.truncation='middle' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$(($BATCH_SIZE * $NUM_GPUS)) \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.log_val_generations=5 \
    trainer.critic_warmup=0 \
    trainer.logger=wandb \
    trainer.project_name='AmazonReviews_Books_grpo' \
    trainer.experiment_name=GRPO-Qwen3-4B-${DATA_NAME} \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.resume_mode=disable \
    trainer.val_before_train=False \
    trainer.default_local_dir=results/ \
    trainer.default_hdfs_dir= \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.total_epochs=2 $@ 2>&1 | tee grpo.log


latest_step=$(cat results/latest_checkpointed_iteration.txt)
echo "Merging checkpoint from step: $latest_step"
python scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir results/global_step_${latest_step}/actor \
    --target_dir ckpt/GRPO_${DATA_NAME}_step${latest_step}

set -x
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_MODE="offline"
export VLLM_USE_V1=0

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/grpo_example.yaml \
    data.train_files=/path/to/stepgrpo_dataset_train.jsonl \
    data.val_files=/path/to/stepgrpo_dataset_valid.jsonl \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.enable_chunked_prefill=false \
    trainer.experiment_name=qwen2_5_vl_7b_stepgrpo \
    trainer.n_gpus_per_node=8

#export CUDA_VISIBLE_DEVICES=0,1,2,3
#
#python -m llamafactory.launcher \
#    --config-path examples/multimodal_stepgrpo_config.yaml \
#    --dataset webqa,mmqa \
#    --template qwen2_vl \
#    --model_name Qwen/Qwen2.5-VL-7B-Instruct \
#    --output_dir ./checkpoints/stepwise_multimodal_router \
#    --reward_model_type stepwise_multimodal \
#    --use_stepwise_rewards True \
#    --information_gain_weight 0.3 \
#    --redundancy_penalty_weight 0.2 \
#    --routing_accuracy_weight 0.3 \
#    --answer_quality_weight 0.2
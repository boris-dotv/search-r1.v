#!/bin/bash
#SBATCH --job-name=Search-R1-PPO
#SBATCH --partition=NV_H100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G                           
#SBATCH --time=72:00:00
#SBATCH --output=/public_hw/share/cit_ztyu/cz/Search-R1/logs/%j_ppo_train.log
#SBATCH --error=/public_hw/share/cit_ztyu/cz/Search-R1/logs/%j_ppo_train.err

PROJECT_ROOT=/public_hw/share/cit_ztyu/cz/Search-R1
mkdir -p $PROJECT_ROOT/logs
mkdir -p $PROJECT_ROOT/wandb_offline

# 检索服务日志
RETRIEVER_LOG=$PROJECT_ROOT/logs/${SLURM_JOB_ID}_retriever.log
RETRIEVER_PID_FILE=$PROJECT_ROOT/logs/${SLURM_JOB_ID}_retriever.pid

cleanup() {
    echo "===== 训练结束/中断，开始关闭检索服务 ====="
    if [ -f $RETRIEVER_PID_FILE ]; then
        RETRIEVER_PID=$(cat $RETRIEVER_PID_FILE)
        if ps -p $RETRIEVER_PID > /dev/null; then
            kill $RETRIEVER_PID
            echo "检索服务进程 $RETRIEVER_PID 已关闭"
        fi
        rm -f $RETRIEVER_PID_FILE
    fi
    echo "===== 清理完成 ====="
}
trap cleanup EXIT

eval "$(conda shell.bash hook)"

# ===================== 第一步：启动检索服务 (保持不变) =====================
echo "===== 第一步：启动检索服务 ====="
cd $PROJECT_ROOT || exit 1

export HF_HOME=$PROJECT_ROOT/hf_cache
export HUGGINGFACE_HUB_CACHE=$PROJECT_ROOT/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_SYMLINKS=True

file_path=$PROJECT_ROOT/indexing_corpus
index_file=$file_path/e5_HNSW64.index
corpus_file=$file_path/wiki-18.jsonl
retriever_name=e5
retriever_path=$PROJECT_ROOT/models/e5-base-v2

conda activate retriever || exit 1

echo "启动检索服务，日志保存到 $RETRIEVER_LOG"
export OMP_NUM_THREADS=32 
export CUDA_VISIBLE_DEVICES=1 

python search_r1/search/retrieval_server.py \
    --index_path $index_file \
    --corpus_path $corpus_file \
    --topk 3 \
    --retriever_name $retriever_name \
    --retriever_model $retriever_path > $RETRIEVER_LOG 2>&1 &

RETRIEVER_PID=$!
echo $RETRIEVER_PID > $RETRIEVER_PID_FILE
echo "检索服务PID: $RETRIEVER_PID"

# ===================== 第二步：等待检索服务就绪 (保持不变) =====================
echo "===== 第二步：等待检索服务就绪 ====="
RETRIEVER_URL="http://127.0.0.1:8009/retrieve"
MAX_WAIT=600
WAIT_INTERVAL=10
WAIT_COUNT=0

while true; do
    HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST $RETRIEVER_URL \
        -H "Content-Type: application/json" \
        -d '{"queries": ["test"], "topk": 1, "return_scores": true}')
    
    if [ "$HTTP_STATUS" -eq 200 ]; then
        echo "检索服务就绪！"
        break
    fi

    WAIT_COUNT=$((WAIT_COUNT + 1))
    if [ $WAIT_COUNT -ge $((MAX_WAIT / WAIT_INTERVAL)) ]; then
        echo "等待检索服务超时！"
        exit 1
    fi
    sleep $WAIT_INTERVAL
done

# ===================== 第三步：启动 PPO 训练 =====================
echo "===== 第三步：启动 PPO 训练 ====="
conda deactivate
conda activate searchr1 || exit 1

# 注意：这里只用 2 张卡
export CUDA_VISIBLE_DEVICES=0,1
export DATA_DIR='data/nq_search'
export TRAIN_DATA_DIR=$DATA_DIR
export TEST_DATA_DIR=$DATA_DIR
export WAND_PROJECT='Search-R1'
export WANDB_MODE=offline
export WANDB_DIR=$PROJECT_ROOT/wandb_offline

# 你的 Base Model 路径
export BASE_MODEL=/public_hw/share/cit_ztyu/cz/models/Qwen2.5-3B-Instruct
# 改名，方便区分
export EXPERIMENT_NAME=nq-search-r1-ppo-qwen2.5-3b-instruct-em

export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export MASTER_PORT=$(shuf -i 20000-60000 -n 1)

echo "训练主端口: $MASTER_PORT"
ray stop --force > /dev/null 2>&1
export RAY_TEMP_DIR=$PROJECT_ROOT/tmp_ray
mkdir -p $RAY_TEMP_DIR

# PPO 启动命令
HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$TRAIN_DATA_DIR/train.parquet \
    data.val_files=$TEST_DATA_DIR/test.parquet \
    data.train_data_num=null \
    data.val_data_num=256 \
    data.train_batch_size=256 \
    data.val_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    data.max_start_length=2048 \
    data.max_obs_length=2048 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=gae \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=32 \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.grad_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=128 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size=128 \
    actor_rollout_ref.ref.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.kl_loss_coef=0.005 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    critic.optim.lr=5e-6 \
    critic.model.use_remove_padding=True \
    critic.optim.lr_warmup_steps_ratio=0.015 \
    critic.model.path=$BASE_MODEL \
    critic.model.enable_gradient_checkpointing=true \
    critic.ppo_micro_batch_size=32 \
    critic.model.fsdp_config.param_offload=false \
    critic.model.fsdp_config.grad_offload=false \
    critic.model.fsdp_config.optimizer_offload=true \
    algorithm.no_think_rl=false \
    trainer.logger=['wandb'] \
    +trainer.val_only=false \
    +trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=100 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=15 \
    trainer.total_training_steps=1005 \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    max_turns=2 \
    retriever.url=$RETRIEVER_URL \
    retriever.topk=3 \
    2>&1 | tee $PROJECT_ROOT/logs/${SLURM_JOB_ID}_${EXPERIMENT_NAME}.log
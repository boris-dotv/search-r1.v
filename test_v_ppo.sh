#!/bin/bash
#SBATCH --job-name=Eval-Search-R1-Native
#SBATCH --partition=NV_H100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=./logs/%j_eval_native.log
#SBATCH --error=./logs/%j_eval_native.err

# ================= 0. 核心配置 =================
ENVS_DIR="/public_hw/home/cit_zhishuliu/miniconda3/envs"
PY_RETRIEVER="$ENVS_DIR/retriever/bin/python"
PY_EVAL="$ENVS_DIR/searchr1/bin/python"

PROJECT_ROOT="/public_hw/share/cit_ztyu/cz/Search-R1"
cd $PROJECT_ROOT || exit 1

# 模型路径
CHECKPOINT_PATH="$PROJECT_ROOT/verl_checkpoints/nq-search-r1-ppo-qwen2.5-3b-instruct-em/actor/global_step_150"
TEST_DATA_PATH="$PROJECT_ROOT/data/nq_search/test.parquet"

# 检索资源
INDEX_FILE="$PROJECT_ROOT/indexing_corpus/e5_HNSW64.index"
CORPUS_FILE="$PROJECT_ROOT/indexing_corpus/wiki-18.jsonl"
RETRIEVER_MODEL="$PROJECT_ROOT/models/e5-base-v2"

# 端口配置
RETRIEVER_PORT=$(shuf -i 40001-50000 -n 1)
RETRIEVER_URL="http://127.0.0.1:$RETRIEVER_PORT/retrieve"

# CUDA 环境 (关键优化：解决显存碎片化)
export CUDA_HOME=/public_hw/software/compiler/cuda/cuda-12.8.1
export PATH=$CUDA_HOME/bin:$PATH
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$PROJECT_ROOT/logs"

# ================= 1. 自动清理机制 =================
RETRIEVER_PID=""
cleanup() {
    if [ -n "$RETRIEVER_PID" ]; then
        echo ">> [Cleanup] Closing Retriever (PID: $RETRIEVER_PID)..."
        kill $RETRIEVER_PID 2>/dev/null
    fi
}
trap cleanup EXIT SIGINT SIGTERM

# ================= 2. 启动检索服务 =================
echo "========================================================"
echo ">> [1/2] 启动检索服务 (Port: $RETRIEVER_PORT)..."
echo "========================================================"

export CUDA_VISIBLE_DEVICES=0

nohup $PY_RETRIEVER search_r1/search/retrieval_server.py \
    --index_path "$INDEX_FILE" \
    --corpus_path "$CORPUS_FILE" \
    --retriever_name e5 \
    --retriever_model "$RETRIEVER_MODEL" \
    --port $RETRIEVER_PORT \
    --topk 3 > "$PROJECT_ROOT/logs/retriever_eval_${SLURM_JOB_ID}.log" 2>&1 &

RETRIEVER_PID=$!
echo ">> Retriever PID: $RETRIEVER_PID"

echo ">> Waiting for Retriever..."
for i in {1..60}; do
    if curl -s "http://127.0.0.1:$RETRIEVER_PORT/retrieve" > /dev/null || \
       curl -o /dev/null -s -w "%{http_code}" "http://127.0.0.1:$RETRIEVER_PORT/retrieve" | grep -q "405"; then
        echo ">> Retriever Ready!"
        break
    fi
    sleep 2
    echo -n "."
done
echo ""

# ================= 3. 启动评估 =================
echo "========================================================"
echo ">> [2/2] 启动 Verl Native Evaluation (1 GPU, Low Batch)..."
echo "========================================================"

# 关键参数调整：
# 1. ppo_mini_batch_size=4, ppo_micro_batch_size=2 : 极大幅度降低单次前向传播的显存需求
# 2. gpu_memory_utilization=0.5 : 降低 vLLM 显存占用，给 PyTorch 留更多空间
# 3. val_batch_size=16 : 稍微降低验证总批次大小，虽然对 OOM 影响不如 micro 明显，但更稳妥

$PY_EVAL -m verl.trainer.main_ppo \
    data.train_files="$TEST_DATA_PATH" \
    data.val_files="$TEST_DATA_PATH" \
    data.val_batch_size=16 \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path="$CHECKPOINT_PATH" \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=0.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    critic.model.path="$CHECKPOINT_PATH" \
    critic.ppo_mini_batch_size=4 \
    critic.ppo_micro_batch_size=2 \
    critic.optim.lr=0 \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.logger=['console'] \
    +trainer.val_only=true \
    retriever.url="$RETRIEVER_URL" \
    retriever.topk=3

echo "========================================================"
echo ">> 评估结束."
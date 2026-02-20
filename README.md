# Search-R1: Train your LLMs to reason and call a search engine with reinforcement learning


Built upon [veRL](https://github.com/volcengine/verl), Search-R1 extends the ideas of **DeepSeek-R1(-Zero)** by incorporating interleaved search engine access and provides a fully open-source RL training pipeline. It serves as an alternative and open solution to **OpenAI DeepResearch**, enabling research and development in tool-augmented LLM reasoning.


Paper: [link1](https://arxiv.org/pdf/2503.09516), [link2](https://arxiv.org/abs/2505.15117); Model and data: [link](https://huggingface.co/collections/PeterJinGo/search-r1-67d1a021202731cb065740f5); Twitter thread: [link](https://x.com/BowenJin13/status/1895544294473109889); Full experiment log: [prelim](https://wandb.ai/peterjin/Search-R1-open); [v0.1](https://wandb.ai/peterjin/Search-R1-nq_hotpotqa_train); [v0.2](https://wandb.ai/peterjin/Search-R1-v0.2); [v0.3](https://wandb.ai/peterjin/Search-R1-v0.3). Details about these logs and methods can be find [here](https://github.com/PeterGriffinJin/Search-R1/blob/main/docs/experiment_log.md).



## Installation

### Search-r1 environment
```bash
conda create -n searchr1 python=3.9
conda activate searchr1
# install torch [or you can skip this step and let vllm to install the correct version for you]
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install vllm==0.6.3 -i https://pypi.tuna.tsinghua.edu.cn/simple # or you can install 0.5.4, 0.4.2 and 0.3.1
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install ./flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp39-cp39-linux_x86_64.whl
pip install wandb -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Retriever environment
If you would like to call a local retriever as the search engine, you can install the environment as follows. (We recommend using a seperate environment.)
```bash
conda create -n retriever python=3.10
conda activate retriever

pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install transformers datasets pyserini -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install faiss-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install uvicorn fastapi -i https://pypi.tuna.tsinghua.edu.cn/simple
```


## Quick start

Train a reasoning + search LLM on NQ dataset with e5 as the retriever and wikipedia as the corpus.

(1) Download the indexing and corpus. 这个方法不稳定.
```bash
# nohup ./download_index.sh > download_task.log 2>&1 &
# ps -ef | grep download.py
# pkill -f download.py
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_SYMLINKS=True
export HF_HOME=/public_hw/share/cit_ztyu/cz/Search-R1/hf_cache
export HUGGINGFACE_HUB_CACHE=/public_hw/share/cit_ztyu/cz/Search-R1/hf_cache

save_path=/public_hw/share/cit_ztyu/cz/Search-R1/indexing_corpus
python scripts/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz
```

手动构建 e5_Flat.index 的方法:
```bash
# 构建索引
conda activate retriever

# --- 关键：再次 export 同样的缓存路径 ---
export HF_HOME=/public_hw/share/cit_ztyu/cz/Search-R1/hf_cache
export HUGGINGFACE_HUB_CACHE=/public_hw/share/cit_ztyu/cz/Search-R1/hf_cache

# 运行你的构建命令
export CUDA_VISIBLE_DEVICES=0
python search_r1/search/index_builder.py \
    --retrieval_method e5 \
    --model_path /public_hw/share/cit_ztyu/cz/Search-R1/models/e5-base-v2 \
    --corpus_path example/corpus.jsonl \
    --save_dir example/mini_index \
    --max_length 256 \
    --batch_size 32 \
    --use_fp16 \
    --faiss_type Flat \
    --save_embedding
```


我测试的稳定下载方法: 
```bash
cd /public_hw/share/cit_ztyu/cz/Search-R1/indexing_corpus/
# 带浏览器请求头+重试策略
# 下载 part_aa
nohup wget -O part_aa \
--user-agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36" \
--tries=20 \
--waitretry=5 \
https://hf-mirror.com/datasets/PeterJinGo/wiki-18-e5-index-HNSW64/resolve/main/part_aa > aa_hnsw.log 2>&1 &

# 下载 part_ab
nohup wget -O part_ab \
--user-agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36" \
--tries=20 \
--waitretry=5 \
https://hf-mirror.com/datasets/PeterJinGo/wiki-18-e5-index-HNSW64/resolve/main/part_ab > ab_hnsw.log 2>&1 &

nohup wget -c -O wiki-18.jsonl.gz https://hf-mirror.com/datasets/PeterJinGo/wiki-18-corpus/resolve/main/wiki-18.jsonl.gz > corpus.log 2>&1 &
# 检查
ls -lh
nohup bash -c "cat part_aa part_ab > e5_HNSW64.index && gzip -d wiki-18.jsonl.gz" > cat.log 2>&1 &
```



(2) Process the NQ dataset.

首先下载数据集:
```bash
# 1. 配置 HF 环境变量（确保国内镜像加速）
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/public_hw/share/cit_ztyu/cz/Search-R1/hf_cache

# 2. 后台加载 NQ 子集并保存到指定目录
nohup python -c "
import datasets
import os

# 配置路径（确保创建 nq/ 目录）
dataset_repo = 'RUC-NLPIR/FlashRAG_datasets'
target_root = '/public_hw/share/cit_ztyu/cz/Search-R1/data/FlashRAG_datasets'
nq_target_dir = os.path.join(target_root, 'nq')  # 手动创建 nq/ 子目录

# 1. 创建 nq/ 目录（若不存在）
os.makedirs(nq_target_dir, exist_ok=True)
print(f'已创建 nq 目录：{nq_target_dir}')

# 2. 直接加载 NQ 子集（datasets 库自动处理下载/缓存，绕开目录匹配问题）
print('开始加载 NQ 数据集...')
nq_dataset = datasets.load_dataset(dataset_repo, 'nq')

# 3. 保存核心数据文件（train.jsonl + test.jsonl，nq_search.py 必需）
print('开始保存 NQ 训练集和测试集...')
train_jsonl_path = os.path.join(nq_target_dir, 'train.jsonl')
test_jsonl_path = os.path.join(nq_target_dir, 'test.jsonl')

nq_dataset['train'].to_json(train_jsonl_path)
nq_dataset['test'].to_json(test_jsonl_path)

print(f'NQ 数据集保存完成！')
print(f'训练集：{train_jsonl_path}')
print(f'测试集：{test_jsonl_path}')
" > download_flashrag_nq_create_dir.log 2>&1 &
```

随后处理数据:

```bash
python scripts/data_process/nq_search.py
```






(3) Launch a local retrieval server.
```bash
netstat -tulpn | grep 8012

srun -p NV_H100 --gres=gpu:1 --cpus-per-task=16 --mem=128G --time=1:00:00 --pty /bin/bash

conda activate retriever
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_SYMLINKS=True
export HF_HOME=/public_hw/share/cit_ztyu/cz/Search-R1/hf_cache
export HUGGINGFACE_HUB_CACHE=/public_hw/share/cit_ztyu/cz/Search-R1/hf_cache

nohup bash retrieval_launch.sh > retrieval_server.log 2>&1 &

curl -X POST "http://127.0.0.1:8009/retrieve" \
-H "Content-Type: application/json" \
-d '{
    "queries": ["What is the capital of China?"],
    "topk": 3,
    "return_scores": true
}'

curl -X POST "http://127.0.0.1:8009/retrieve" \
-H "Content-Type: application/json" \
-d '{
    "queries": ["What is the capital of China?"],
    "topk": 3,
    "return_scores": true
}' \
-o /dev/null -s -w "\n%{time_total}\n"
```

(4) Run RL training (GRPO) with Qwen/Qwen2.5-3B-Instruct.
首先下载模型:
```bash
# 确保镜像加速, 无符号链接
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_SYMLINKS=True
export HF_HOME=/public_hw/share/cit_ztyu/cz/Search-R1/hf_cache

nohup python -c "
from huggingface_hub import snapshot_download

# 模型名称 & 目标目录
repo_id = 'Qwen/Qwen2.5-3B-Instruct'
target_dir = '/public_hw/share/cit_ztyu/cz/models/Qwen2.5-3B-Instruct'

# 核心参数保留，移除多余的 timeout 参数
snapshot_download(
    repo_id=repo_id,
    local_dir=target_dir,
    local_dir_use_symlinks=False,  # 确保下载源文件，无哈希/符号链接
    cache_dir=None,
    force_download=False,  # 断点续传，不重复下载已完成文件
    repo_type='model'  # 明确下载模型，更严谨（可省略，默认即为 model）
)
" > download_qwen_model_hf.log 2>&1 &
```



随后提交训练任务:
```bash
# 1. 清理你自己的 Ray 残留进程（精准，不影响同学）
ps -u cit_zhishuliu | grep ray | grep -v grep | awk '{print $1}' | xargs kill -9 2>/dev/null

# 2. 重新提交作业
cd /public_hw/share/cit_ztyu/cz/Search-R1
sbatch train_v.sh
```



## 训练很慢的问题


查看 wandb 具体配置和数据:
```bash
# "mfu/actor": 0.2060669977044205
# 那次无效的缓慢训练
cat wandb_offline/wandb/offline-run-20260206_115849-9r22pa1l/files/wandb-summary.json
cat wandb_offline/wandb/offline-run-20260206_115849-9r22pa1l/files/config.yaml
tail -n 50 wandb_offline/wandb/offline-run-20260206_115849-9r22pa1l/files/output.log



# 有效的加速训练
cat /public_hw/share/cit_ztyu/cz/Search-R1/wandb_offline/wandb/offline-run-20260207_144224-hcnzd992/files/wandb-summary.json
tail -n 50 /public_hw/share/cit_ztyu/cz/Search-R1/wandb_offline/wandb/offline-run-20260207_144224-hcnzd992/files/output.log

cat /public_hw/share/cit_ztyu/cz/Search-R1/wandb_offline/wandb/offline-run-20260208_103244-6m9dm6kk/files/wandb-summary.json
tail -n 50 /public_hw/share/cit_ztyu/cz/Search-R1/wandb_offline/wandb/offline-run-20260208_103244-6m9dm6kk/files/output.log

cat /public_hw/share/cit_ztyu/cz/Search-R1/wandb_offline/wandb/offline-run-20260208_131831-oqsx4ys9/files/wandb-summary.json
tail -n 50 /public_hw/share/cit_ztyu/cz/Search-R1/wandb_offline/wandb/offline-run-20260208_131831-oqsx4ys9/files/output.log
```



测试检索服务:

```bash
(retriever) cit_zhishuliu@gpuh05:/public_hw/share/cit_ztyu/cz/Search-R1$ bash test_retrieval_perf.sh 
======================================
Performance Testing Script
======================================

[Test 1] Single query performance:
Response time: 11.825183s

real    0m11.845s
user    0m0.005s
sys     0m0.013s

[Test 2] Batch query performance (10 queries):
Response time: 11.537487s

real    0m11.555s
user    0m0.012s
sys     0m0.006s

[Test 3] Large batch query performance (100 queries):
100 queries completed in: 103.878s
Average per query: 1038.78ms
Throughput: 0.96 queries/sec

[Test 4] Training-scale batch (128 queries):
128 queries (1 training batch) completed in: 130.489s
Average per query: 1019.44ms
Throughput: 0.98 queries/sec

======================================
Testing complete!
======================================
```




[更换搜索引擎](https://github.com/PeterGriffinJin/Search-R1/blob/main/docs/retriever.md), 然后非常快, 快了 2000 倍:

```bash
(retriever) cit_zhishuliu@gpuh01:/public_hw/share/cit_ztyu/cz/Search-R1$ bash test_retrieval_perf.sh 
======================================
Performance Testing Script
======================================

[Test 1] Single query performance:
Response time: 0.561613s

real    0m0.575s
user    0m0.000s
sys     0m0.010s

[Test 2] Batch query performance (10 queries):
Response time: 0.126714s

real    0m0.134s
user    0m0.006s
sys     0m0.000s

[Test 3] Large batch query performance (100 queries):
100 queries completed in: 0.345s
Average per query: 3.45ms
Throughput: 290.13 queries/sec

[Test 4] Training-scale batch (128 queries):
128 queries (1 training batch) completed in: 0.065s
Average per query: 0.51ms
Throughput: 1958.58 queries/sec

======================================
Testing complete!
======================================
(retriever) cit_zhishuliu@gpuh01:/public_hw/share/cit_ztyu/cz/Search-R1$ 
```




**权重保存问题, 以及 validation 随机性 / 确定性问题**


## Preliminary results

(1) The base model (llama3.2-3b-base) learns to call the search engine and obtain improved performance.

![llama-3b](public/llama32-3b.png)


(2) The base model (Qwen2.5-7b-base) can learn to conduct multi-turn search engine calling and reasoning with RL.

![multi-turn](public/multi-turn.png)




## Inference
#### You can play with the trained Search-R1 model with your own question.
(1) Launch a local retrieval server.
```bash
srun -p NV_H100 --gres=gpu:1 --cpus-per-task=16 --mem=128G --time=1:00:00 --pty /bin/bash

conda activate retriever
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_SYMLINKS=True
export HF_HOME=/public_hw/share/cit_ztyu/cz/Search-R1/hf_cache
export HUGGINGFACE_HUB_CACHE=/public_hw/share/cit_ztyu/cz/Search-R1/hf_cache

bash retrieval_launch.sh

curl -X POST "http://127.0.0.1:8009/retrieve" \
-H "Content-Type: application/json" \
-d '{
    "queries": ["What is the capital of China?"],
    "topk": 3,
    "return_scores": true
}'
```

(2) Run inference.
```bash
conda activate searchr1
python infer.py
```
You can modify the ```question``` on line 7 to something you're interested in.





## Use your own dataset

### QA data
For each question-answer sample, it should be a dictionary containing the desired content as below:

```
data = {
        "data_source": data_source,
        "prompt": [{
            "role": "user",
            "content": question,
        }],
        "ability": "fact-reasoning",
        "reward_model": {
            "style": "rule",
            "ground_truth": solution
        },
        "extra_info": {
            'split': split,
            'index': idx,
        }
    }
```

You can refer to ```scripts/data_process/nq_search.py``` for a concrete data processing example.





## Features
- Support local sparse retrievers (e.g., BM25). ✔️
- Support local dense retrievers (both flat indexing and ANN indexing) ✔️
- Support google search / bing search / brave search API and others. ✔️
- Support off-the-shelf neural rerankers. ✔️
- Support different RL methods (e.g., PPO, GRPO, reinforce). ✔️
- Support different LLMs (e.g., llama3, Qwen2.5, etc). ✔️

## Acknowledge

Its implementation is built upon [Search-R1](https://github.com/PeterGriffinJin/Search-R1).


## Awesome work powered or inspired by Search-R1

- [DeepResearcher](https://github.com/GAIR-NLP/DeepResearcher): Scaling Deep Research via Reinforcement Learning in Real-world Environments. [![[code]](https://img.shields.io/github/stars/GAIR-NLP/DeepResearcher)](https://github.com/GAIR-NLP/DeepResearcher)
- [Multimodal-Search-R1](https://github.com/EvolvingLMMs-Lab/multimodal-search-r1): Incentivizing LMMs to Search. [![[code]](https://img.shields.io/github/stars/EvolvingLMMs-Lab/multimodal-search-r1)](https://github.com/EvolvingLMMs-Lab/multimodal-search-r1)
- [OTC](https://arxiv.org/pdf/2504.14870): Optimal Tool Calls via Reinforcement Learning.
- [ZeroSearch](https://github.com/Alibaba-NLP/ZeroSearch): Incentivize the Search Capability of LLMs without Searching. [![[code]](https://img.shields.io/github/stars/Alibaba-NLP/ZeroSearch)](https://github.com/Alibaba-NLP/ZeroSearch)
- [IKEA](https://github.com/hzy312/knowledge-r1): Reinforced Internal-External Knowledge Synergistic Reasoning for Efficient Adaptive Search Agent. [![[code]](https://img.shields.io/github/stars/hzy312/knowledge-r1)](https://github.com/hzy312/knowledge-r1)
- [Scent of Knowledge](https://arxiv.org/abs/2505.09316): Optimizing Search-Enhanced Reasoning with Information Foraging.
- [AutoRefine](https://www.arxiv.org/pdf/2505.11277): Search and Refine During Think. [![[code]](https://img.shields.io/github/stars/syr-cn/AutoRefine)](https://github.com/syr-cn/AutoRefine)
- [O^2-Searcher](https://arxiv.org/pdf/2505.16582): A Searching-based Agent Model for Open-Domain Open-Ended Question Answering. [![[code]](https://img.shields.io/github/stars/Acade-Mate/O2-Searcher)](https://github.com/Acade-Mate/O2-Searcher)
- [MaskSearch](https://arxiv.org/pdf/2505.20285): A Universal Pre-Training Framework to Enhance Agentic Search Capability. [![[code]](https://img.shields.io/github/stars/Alibaba-NLP/MaskSearch)](https://github.com/Alibaba-NLP/MaskSearch)
- [VRAG-RL](https://arxiv.org/abs/2505.22019): Vision-Perception-Based RAG for Visually Rich Information Understanding. [![[code]](https://img.shields.io/github/stars/Alibaba-NLP/VRAG)](https://github.com/Alibaba-NLP/VRAG)
- [R1-Code-Interpreter](https://arxiv.org/abs/2505.21668): Training LLMs to Reason with Code via SFT and RL. [![[code]](https://img.shields.io/github/stars/yongchao98/R1-Code-Interpreter)](https://github.com/yongchao98/R1-Code-Interpreter)
- [R-Search](https://arxiv.org/abs/2506.04185): Empowering LLM Reasoning with Search via Multi-Reward Reinforcement Learning. [![[code]](https://img.shields.io/github/stars/QingFei1/R-Search)](https://github.com/QingFei1/R-Search)
- [StepSearch](https://arxiv.org/pdf/2505.15107): Igniting LLMs Search Ability via Step-Wise Proximal Policy Optimization. [![[code]](https://img.shields.io/github/stars/Zillwang/StepSearch)](https://github.com/Zillwang/StepSearch)
- [SimpleTIR](https://simpletir.notion.site/report): Stable End-to-End Reinforcement Learning for Multi-Turn Tool-Integrated Reasoning. [![[code]](https://img.shields.io/github/stars/ltzheng/SimpleTIR)](https://github.com/ltzheng/SimpleTIR)
- [Router-R1](https://arxiv.org/pdf/2506.09033): Teaching LLMs Multi-Round Routing and Aggregation via Reinforcement Learning. [![[code]](https://img.shields.io/github/stars/ulab-uiuc/Router-R1)](https://github.com/ulab-uiuc/Router-R1)
- [SkyRL](https://skyrl.readthedocs.io/en/latest/): A Modular Full-stack RL Library for LLMs. [![[code]](https://img.shields.io/github/stars/NovaSky-AI/SkyRL)](https://github.com/NovaSky-AI/SkyRL)
- [ASearcher](https://arxiv.org/abs/2508.07976): Large-Scale RL for Search Agents. [![[code]](https://img.shields.io/github/stars/inclusionAI/ASearcher)](https://github.com/inclusionAI/ASearcher)
- [ParallelSearch](https://www.arxiv.org/abs/2508.09303): Decompose Query and Search Sub-queries in Parallel with RL. [![[code]](https://img.shields.io/github/stars/Tree-Shu-Zhao/ParallelSearch)](https://github.com/Tree-Shu-Zhao/ParallelSearch)
- [AutoTIR](https://arxiv.org/pdf/2507.21836): Autonomous Tools Integrated Reasoning via Reinforcement Learning. [![[code]](https://img.shields.io/github/stars/weiyifan1023/AutoTIR)](https://github.com/weiyifan1023/AutoTIR)
- [verl-tool](https://arxiv.org/pdf/2509.01055): A version of verl to support diverse tool use. [![[code]](https://img.shields.io/github/stars/TIGER-AI-Lab/verl-tool)](https://github.com/TIGER-AI-Lab/verl-tool)
- [Tree-GRPO](https://arxiv.org/abs/2509.21240): Tree Search for LLM Agent Reinforcement Learning. [![[code]](https://img.shields.io/github/stars/AMAP-ML/Tree-GRPO)](https://github.com/AMAP-ML/Tree-GRPO)
- [EviNote-RAG](https://arxiv.org/abs/2509.00877): Enhancing RAG Models via Answer-Supportive Evidence Notes. [![[code]](https://img.shields.io/github/stars/Da1yuqin/EviNoteRAG)](https://github.com/Da1yuqin/EviNoteRAG)
- [GlobalRAG](https://arxiv.org/pdf/2510.20548v1): GlobalRAG: Enhancing Global Reasoning in Multi-hop Question Answering via Reinforcement Learning. [![[code]](https://img.shields.io/github/stars/CarnegieBin/GlobalRAG)](https://github.com/CarnegieBin/GlobalRAG)



可以通过 cfg = FullStateDictConfig(offload_to_cpu=False, rank0_only=True) 来解决:

```bash
[36m(WorkerDict pid=814757)[0m /public_hw/home/cit_zhishuliu/miniconda3/envs/searchr1/lib/python3.9/site-packages/torch/distributed/fsdp/fully_sharded_data_parallel.py:689: FutureWarning: FSDP.state_dict_type() and FSDP.set_state_dict_type() are being deprecated. Please use APIs, get_state_dict() and set_state_dict(), which can support different parallelisms, FSDP1, FSDP2, DDP. API doc: https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.get_state_dict .Tutorial: https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html .
[36m(WorkerDict pid=814757)[0m   warnings.warn(
[36m(WorkerDict pid=816513)[0m /public_hw/home/cit_zhishuliu/miniconda3/envs/searchr1/lib/python3.9/site-packages/torch/utils/checkpoint.py:1399: FutureWarning: `torch.cpu.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cpu', args...)` instead.
[36m(WorkerDict pid=816513)[0m   with device_autocast_ctx, torch.cpu.amp.autocast(**cpu_autocast_kwargs), recompute_context:  # type: ignore[attr-defined]
```

可以通过 actor_rollout_ref.ref.fsdp_config.param_offload=false 来解决以下问题:
```text
[36m(WorkerDict pid=2370037)[0m /public_hw/home/cit_zhishuliu/miniconda3/envs/searchr1/lib/python3.9/site-packages/torch/utils/checkpoint.py:1399: FutureWarning: `torch.cpu.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cpu', args...)` instead.
[36m(WorkerDict pid=2370037)[0m   with device_autocast_ctx, torch.cpu.amp.autocast(**cpu_autocast_kwargs), recompute_context:  # type: ignore[attr-defined]
[36m(WorkerDict pid=2374935)[0m /public_hw/home/cit_zhishuliu/miniconda3/envs/searchr1/lib/python3.9/site-packages/torch/distributed/fsdp/fully_sharded_data_parallel.py:689: FutureWarning: FSDP.state_dict_type() and FSDP.set_state_dict_type() are being deprecated. Please use APIs, get_state_dict() and set_state_dict(), which can support different parallelisms, FSDP1, FSDP2, DDP. API doc: https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.get_state_dict .Tutorial: https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html .
[36m(WorkerDict pid=2374935)[0m   warnings.warn(
[36m(WorkerDict pid=2370037)[0m /public_hw/home/cit_zhishuliu/miniconda3/envs/searchr1/lib/python3.9/site-packages/torch/distributed/fsdp/fully_sharded_data_parallel.py:689: FutureWarning: FSDP.state_dict_type() and FSDP.set_state_dict_type() are being deprecated. Please use APIs, get_state_dict() and set_state_dict(), which can support different parallelisms, FSDP1, FSDP2, DDP. API doc: https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.get_state_dict .Tutorial: https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html .
[36m(WorkerDict pid=2370037)[0m   warnings.warn(
[36m(WorkerDict pid=2370037)[0m Saving actor checkpoint to verl_checkpoints/nq-search-r1-grpo-qwen2.5-3b-instruct-em/actor/global_step_2
[33m(raylet)[0m A worker died or was killed while executing a task by an unexpected system error. To troubleshoot the problem, check the logs for the dead worker. Lease ID: 000000003b5c1359eaba804df7a6f676b30fdb2ab6a0d42a4149d273495e02f9 Worker ID: a5f83b036b37b94fb115881b5b9d9d1c32ccd636395a830f223c96ee Node ID: 2b49af218cdef8583384e193e4185785c71b735618a63da4d54ac276 Worker IP address: 172.18.103.5 Worker port: 33979 Worker PID: 2370037 Worker exit type: SYSTEM_ERROR Worker exit detail: Worker unexpectedly exits with a connection error code 2. End of file. There are some potential root causes. (1) The process is killed by SIGKILL by OOM killer due to high memory usage. (2) ray stop --force is called. (3) The worker is crashed unexpectedly due to SIGSEGV or other unexpected errors.
Error executing job with overrides: ['data.train_files=data/nq_search/train.parquet', 'data.val_files=data/nq_search/test.parquet', 'data.train_data_num=null', 'data.val_data_num=256', 'data.train_batch_size=64', 'data.val_batch_size=64', 'data.max_prompt_length=4096', 'data.max_response_length=500', 'data.max_start_length=2048', 'data.max_obs_length=2048', 'data.shuffle_train_dataloader=True', 'algorithm.adv_estimator=grpo', 'actor_rollout_ref.model.path=/public_hw/share/cit_ztyu/cz/models/Qwen2.5-3B-Instruct', 'actor_rollout_ref.model.enable_gradient_checkpointing=true', 'actor_rollout_ref.model.use_remove_padding=True', 'actor_rollout_ref.actor.optim.lr=1e-6', 'actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285', 'actor_rollout_ref.actor.use_kl_loss=true', 'actor_rollout_ref.actor.ppo_mini_batch_size=32', 'actor_rollout_ref.actor.ppo_micro_batch_size=8', 'actor_rollout_ref.actor.fsdp_config.param_offload=false', 'actor_rollout_ref.actor.fsdp_config.grad_offload=false', 'actor_rollout_ref.actor.fsdp_config.optimizer_offload=false', 'actor_rollout_ref.rollout.log_prob_micro_batch_size=64', 'actor_rollout_ref.rollout.tensor_model_parallel_size=1', 'actor_rollout_ref.rollout.name=vllm', 'actor_rollout_ref.rollout.gpu_memory_utilization=0.45', 'actor_rollout_ref.ref.log_prob_micro_batch_size=64', 'actor_rollout_ref.ref.fsdp_config.param_offload=true', 'actor_rollout_ref.actor.kl_loss_coef=0.001', 'actor_rollout_ref.actor.kl_loss_type=low_var_kl', 'algorithm.no_think_rl=false', 'actor_rollout_ref.rollout.n_agent=4', 'actor_rollout_ref.rollout.temperature=1', 'actor_rollout_ref.actor.state_masking=true', 'trainer.logger=[wandb]', '+trainer.val_only=false', '+trainer.val_before_train=false', 'trainer.default_hdfs_dir=null', 'trainer.n_gpus_per_node=2', 'trainer.nnodes=1', 'trainer.save_freq=1', 'trainer.test_freq=100', 'trainer.project_name=Search-R1', 'trainer.experiment_name=nq-search-r1-grpo-qwen2.5-3b-instruct-em', 'trainer.total_epochs=15', 'trainer.total_training_steps=1005', 'trainer.default_hdfs_dir=null', 'trainer.default_local_dir=verl_checkpoints/nq-search-r1-grpo-qwen2.5-3b-instruct-em', 'max_turns=2', 'retriever.url=http://127.0.0.1:8009/retrieve', 'retriever.topk=3']
Traceback (most recent call last):
  File "/public_hw/home/cit_zhishuliu/miniconda3/envs/searchr1/lib/python3.9/runpy.py", line 197, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/public_hw/home/cit_zhishuliu/miniconda3/envs/searchr1/lib/python3.9/runpy.py", line 87, in _run_code
    exec(code, run_globals)
  File "/public_hw/share/cit_ztyu/cz/Search-R1/verl/trainer/main_ppo.py", line 207, in <module>
    main()
  File "/public_hw/home/cit_zhishuliu/miniconda3/envs/searchr1/lib/python3.9/site-packages/hydra/main.py", line 94, in decorated_main
    _run_hydra(
  File "/public_hw/home/cit_zhishuliu/miniconda3/envs/searchr1/lib/python3.9/site-packages/hydra/_internal/utils.py", line 394, in _run_hydra
    _run_app(
  File "/public_hw/home/cit_zhishuliu/miniconda3/envs/searchr1/lib/python3.9/site-packages/hydra/_internal/utils.py", line 457, in _run_app
    run_and_report(
  File "/public_hw/home/cit_zhishuliu/miniconda3/envs/searchr1/lib/python3.9/site-packages/hydra/_internal/utils.py", line 223, in run_and_report
    raise ex
  File "/public_hw/home/cit_zhishuliu/miniconda3/envs/searchr1/lib/python3.9/site-packages/hydra/_internal/utils.py", line 220, in run_and_report
    return func()
  File "/public_hw/home/cit_zhishuliu/miniconda3/envs/searchr1/lib/python3.9/site-packages/hydra/_internal/utils.py", line 458, in <lambda>
    lambda: hydra.run(
  File "/public_hw/home/cit_zhishuliu/miniconda3/envs/searchr1/lib/python3.9/site-packages/hydra/_internal/hydra.py", line 132, in run
    _ = ret.return_value
  File "/public_hw/home/cit_zhishuliu/miniconda3/envs/searchr1/lib/python3.9/site-packages/hydra/core/utils.py", line 260, in return_value
    raise self._return_value
  File "/public_hw/home/cit_zhishuliu/miniconda3/envs/searchr1/lib/python3.9/site-packages/hydra/core/utils.py", line 186, in run_job
    ret.return_value = task_function(task_cfg)
  File "/public_hw/share/cit_ztyu/cz/Search-R1/verl/trainer/main_ppo.py", line 114, in main
    ray.get(main_task.remote(config))
  File "/public_hw/home/cit_zhishuliu/miniconda3/envs/searchr1/lib/python3.9/site-packages/ray/_private/auto_init_hook.py", line 22, in auto_init_wrapper
    return fn(*args, **kwargs)
  File "/public_hw/home/cit_zhishuliu/miniconda3/envs/searchr1/lib/python3.9/site-packages/ray/_private/client_mode_hook.py", line 104, in wrapper
    return func(*args, **kwargs)
  File "/public_hw/home/cit_zhishuliu/miniconda3/envs/searchr1/lib/python3.9/site-packages/ray/_private/worker.py", line 2961, in get
    values, debugger_breakpoint = worker.get_objects(
  File "/public_hw/home/cit_zhishuliu/miniconda3/envs/searchr1/lib/python3.9/site-packages/ray/_private/worker.py", line 1026, in get_objects
    raise value.as_instanceof_cause()
ray.exceptions.RayTaskError(ActorDiedError): [36mray::main_task()[39m (pid=2368471, ip=172.18.103.5)
  File "/public_hw/share/cit_ztyu/cz/Search-R1/verl/trainer/main_ppo.py", line 203, in main_task
    trainer.fit()
  File "/public_hw/share/cit_ztyu/cz/Search-R1/verl/trainer/ppo/ray_trainer.py", line 835, in fit
    self._save_checkpoint()
  File "/public_hw/share/cit_ztyu/cz/Search-R1/verl/trainer/ppo/ray_trainer.py", line 629, in _save_checkpoint
    self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path)
  File "/public_hw/share/cit_ztyu/cz/Search-R1/verl/single_controller/ray/base.py", line 42, in func
    output = ray.get(output)
ray.exceptions.ActorDiedError: The actor died unexpectedly before finishing this task.
	class_name: create_colocated_worker_cls.<locals>.WorkerDict
	actor_id: 944845519790ea751c6613aa01000000
	pid: 2370037
	name: UpWnbxWorkerDict_0:0
	namespace: d4367eba-c8ee-422d-b133-f7de44c99a15
	ip: 172.18.103.5
The actor is dead because its worker process has died. Worker exit type: SYSTEM_ERROR Worker exit detail: Worker unexpectedly exits with a connection error code 2. End of file. There are some potential root causes. (1) The process is killed by SIGKILL by OOM killer due to high memory usage. (2) ray stop --force is called. (3) The worker is crashed unexpectedly due to SIGSEGV or other unexpected errors.
```


```bash
git add .
git commit -m "Initial commit: Upload Search-R1 project"
git remote add origin https://github.com/boris-dotv/search-r1.v.git
git branch -M main
git push -u origin main

ssh-keygen -t ed25519 -C "1322553126@qq.com"
cat ~/.ssh/id_ed25519.pub
git remote set-url origin git@github.com:boris-dotv/search-r1.v.git
git remote -v

git add .
git commit -m "updated cmds in README"
git push -u origin main
git push -u origin main -f
```
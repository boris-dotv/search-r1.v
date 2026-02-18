import json
import os
import warnings
from typing import List, Dict, Optional
import argparse
import time  # 引入时间模块

import faiss
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModel
from tqdm import tqdm
import datasets

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

def load_corpus(corpus_path: str):
    print(f"Loading corpus from {corpus_path}...")
    t0 = time.time()
    corpus = datasets.load_dataset(
        'json', 
        data_files=corpus_path,
        split="train",
        num_proc=4
    )
    print(f"Corpus loaded in {time.time() - t0:.2f}s")
    return corpus

def load_docs(corpus, doc_idxs):
    """
    [Optimized] 优化的批量文档加载函数
    利用 HuggingFace Dataset 的切片特性，一次 IO 读取多行
    """
    # 1. 确保索引是整数列表
    indices = [int(i) for i in doc_idxs]
    
    # 2. 批量切片读取 (这是最关键的一步，速度比循环快 100 倍)
    # dataset[list_of_indices] 会返回一个字典: {'text': [t1, t2...], 'title': [t1, t2...]}
    batch_data = corpus[indices]
    
    # 3. 将列式存储 (Dict of Lists) 转为行式存储 (List of Dicts)
    # 这一步纯 CPU 内存操作，极快
    keys = batch_data.keys()
    results = [dict(zip(keys, vals)) for vals in zip(*batch_data.values())]
    
    return results

def load_model(model_path: str, use_fp16: bool = False):
    print(f"Loading model from {model_path}...")
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model.cuda()
    if use_fp16: 
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    return model, tokenizer

def pooling(pooler_output, last_hidden_state, attention_mask=None, pooling_method="mean"):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError("Pooling method not implemented!")

class Encoder:
    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16):
        self.model_name = model_name
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16

        self.model, self.tokenizer = load_model(model_path=model_path, use_fp16=use_fp16)
        self.model.eval()

    @torch.no_grad()
    def encode(self, query_list: List[str], is_query=True) -> np.ndarray:
        """
        [Deep Profiling] 深入分析编码环节
        """
        t_start = time.time()

        if isinstance(query_list, str):
            query_list = [query_list]

        # 1. 文本预处理
        if "e5" in self.model_name.lower():
            if is_query:
                query_list = [f"query: {query}" for query in query_list]
            else:
                query_list = [f"passage: {query}" for query in query_list]
        
        # 2. Tokenization
        t_tok_start = time.time()
        inputs = self.tokenizer(query_list,
                                max_length=self.max_length,
                                padding=True,
                                truncation=True,
                                return_tensors="pt"
                                )
        t_tok = time.time() - t_tok_start

        # 3. GPU Transfer
        t_cuda_start = time.time()
        inputs = {k: v.cuda() for k, v in inputs.items()}
        t_cuda = time.time() - t_cuda_start

        # 4. Model Inference
        t_infer_start = time.time()
        if "T5" in type(self.model).__name__:
            decoder_input_ids = torch.zeros((inputs['input_ids'].shape[0], 1), dtype=torch.long).to(inputs['input_ids'].device)
            output = self.model(**inputs, decoder_input_ids=decoder_input_ids, return_dict=True)
            query_emb = output.last_hidden_state[:, 0, :]
        else:
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(output.pooler_output, output.last_hidden_state, inputs['attention_mask'], self.pooling_method)
            if "dpr" not in self.model_name.lower():
                query_emb = torch.nn.functional.normalize(query_emb, dim=-1)
        
        # Ensure async CUDA operations are done for accurate timing
        torch.cuda.synchronize() 
        t_infer = time.time() - t_infer_start

        # 5. CPU Transfer & Numpy
        t_cpu_start = time.time()
        query_emb = query_emb.detach().cpu().numpy()
        query_emb = query_emb.astype(np.float32, order="C")
        t_cpu = time.time() - t_cpu_start
        
        torch.cuda.empty_cache()

        t_total = time.time() - t_start
        
        # 打印编码详情
        if t_total > 0.05: # 只有当编码时间显著时才打印
            print(f"  [Encode Detail] Batch: {len(query_list)}, Total: {t_total:.4f}s")
            print(f"  --> Tokenize: {t_tok*1000:.2f}ms | ToGPU: {t_cuda*1000:.2f}ms | Inference: {t_infer*1000:.2f}ms | ToCPU: {t_cpu*1000:.2f}ms")

        return query_emb

class BaseRetriever:
    def __init__(self, config):
        self.config = config
        self.retrieval_method = config.retrieval_method
        self.topk = config.retrieval_topk
        self.index_path = config.index_path
        self.corpus_path = config.corpus_path

    def batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        raise NotImplementedError

class BM25Retriever(BaseRetriever):
    # 省略 BM25 部分，因为你用的是 DenseRetriever
    def __init__(self, config):
        super().__init__(config)
        pass 
    def batch_search(self, query_list, num=None, return_score=False):
        return []

class DenseRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        
        # === 核心修改：强制设置 CPU 线程数 ===
        # 针对你的 96 核机器，设置 32 或 64 都可以，HNSW 极度依赖这个
        if not config.faiss_gpu:
            target_threads = 32
            faiss.omp_set_num_threads(target_threads)
            print(f"[Config] CPU Mode detected. Force FAISS OMP threads to: {faiss.omp_get_max_threads()}")
        # ==================================

        print(f"Loading FAISS index from {self.index_path}...")
        self.index = faiss.read_index(self.index_path)
        
        if config.faiss_gpu:
            print("Moving FAISS index to GPU...")
            res = faiss.StandardGpuResources()
            res.setTempMemory(0)  
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True  
            co.shard = True  
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index, co)  

        self.corpus = load_corpus(self.corpus_path)
        self.encoder = Encoder(
            model_name = self.retrieval_method,
            model_path = config.retrieval_model_path,
            pooling_method = config.retrieval_pooling_method,
            max_length = config.retrieval_query_max_length,
            use_fp16 = config.retrieval_use_fp16
        )
        self.topk = config.retrieval_topk
        self.batch_size = config.retrieval_batch_size

    def batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.topk
        
        results = []
        scores = []
        
        # 阶段耗时统计
        time_stats = {
            "encode": 0.0,
            "search": 0.0,
            "load": 0.0
        }
        
        print(f"\n[Batch Start] Processing {len(query_list)} queries...")
        
        for start_idx in range(0, len(query_list), self.batch_size):
            query_batch = query_list[start_idx:start_idx + self.batch_size]
            
            # 1. Encode
            t0 = time.time()
            batch_emb = self.encoder.encode(query_batch)
            time_stats["encode"] += (time.time() - t0)
            
            # 2. FAISS Search
            t0 = time.time()
            batch_scores, batch_idxs = self.index.search(batch_emb, k=num)
            torch.cuda.synchronize() # 确保搜索完成
            time_stats["search"] += (time.time() - t0)
            
            batch_scores = batch_scores.tolist()
            batch_idxs = batch_idxs.tolist()

            # 3. Load Docs
            t0 = time.time()
            flat_idxs = sum(batch_idxs, [])
            # 调用我们修改过的详细打点 load_docs
            batch_results = load_docs(self.corpus, flat_idxs)
            # 重组
            batch_results = [batch_results[i*num : (i+1)*num] for i in range(len(batch_idxs))]
            time_stats["load"] += (time.time() - t0)
            
            results.extend(batch_results)
            scores.extend(batch_scores)
            
            torch.cuda.empty_cache()
            
        print("-" * 50)
        print(f"Batch Summary (Queries: {len(query_list)})")
        print(f"  Total Encode: {time_stats['encode']:.4f}s")
        print(f"  Total Search: {time_stats['search']:.4f}s")
        print(f"  Total DocLoad: {time_stats['load']:.4f}s")
        print("-" * 50)

        if return_score:
            return results, scores
        else:
            return results

def get_retriever(config):
    # 简化：只返回 DenseRetriever 用于测试
    return DenseRetriever(config)

# ... (Config 类和 FastAPI app 部分保持不变，或者直接复制下面的) ...

class Config:
    def __init__(
        self, 
        retrieval_method: str = "bm25", 
        retrieval_topk: int = 10,
        index_path: str = "./index/bm25",
        corpus_path: str = "./data/corpus.jsonl",
        dataset_path: str = "./data",
        data_split: str = "train",
        faiss_gpu: bool = True,
        retrieval_model_path: str = "./model",
        retrieval_pooling_method: str = "mean",
        retrieval_query_max_length: int = 256,
        retrieval_use_fp16: bool = False,
        retrieval_batch_size: int = 128
    ):
        self.retrieval_method = retrieval_method
        self.retrieval_topk = retrieval_topk
        self.index_path = index_path
        self.corpus_path = corpus_path
        self.dataset_path = dataset_path
        self.data_split = data_split
        self.faiss_gpu = faiss_gpu
        self.retrieval_model_path = retrieval_model_path
        self.retrieval_pooling_method = retrieval_pooling_method
        self.retrieval_query_max_length = retrieval_query_max_length
        self.retrieval_use_fp16 = retrieval_use_fp16
        self.retrieval_batch_size = retrieval_batch_size


class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False


app = FastAPI()

@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    t_req_start = time.time()
    
    if not request.topk:
        request.topk = config.retrieval_topk

    batch_result = retriever.batch_search(
        query_list=request.queries,
        num=request.topk,
        return_score=request.return_scores
    )

    if isinstance(batch_result, tuple) and len(batch_result) == 2:
        results, scores = batch_result
    else:
        results = batch_result
        scores = None
    
    resp = []
    for i, single_result in enumerate(results):
        if request.return_scores:
            combined = []
            for doc, score in zip(single_result, scores[i]):
                combined.append({"document": doc, "score": score})
            resp.append(combined)
        else:
            resp.append(single_result)
            
    print(f"[API] Total Request Time: {time.time() - t_req_start:.4f}s")
    return {"result": resp}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", type=str, default="/home/peterjin/mnt/index/wiki-18/e5_Flat.index")
    parser.add_argument("--corpus_path", type=str, default="/home/peterjin/mnt/data/retrieval-corpus/wiki-18.jsonl")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--retriever_name", type=str, default="e5")
    parser.add_argument("--retriever_model", type=str, default="intfloat/e5-base-v2")
    parser.add_argument('--faiss_gpu', action='store_true')

    args = parser.parse_args()
    
    config = Config(
        retrieval_method = args.retriever_name,
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        retrieval_topk=args.topk,
        faiss_gpu=args.faiss_gpu,
        retrieval_model_path=args.retriever_model,
        retrieval_pooling_method="mean",
        retrieval_query_max_length=256,
        retrieval_use_fp16=True,
        retrieval_batch_size=512,
    )

    retriever = get_retriever(config)
    uvicorn.run(app, host="0.0.0.0", port=8009)
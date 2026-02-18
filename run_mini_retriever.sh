#!/bin/bash
# 这里的路径严格对应刚才生成的索引
index_file=example/mini_index/e5_Flat.index
corpus_file=example/corpus.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2

echo "Index File: $index_file"
echo "Corpus File: $corpus_file"

python search_r1/search/retrieval_server.py \
    --index_path $index_file \
    --corpus_path $corpus_file \
    --topk 3 \
    --retriever_name $retriever_name \
    --retriever_model $retriever_path \
    --faiss_gpu

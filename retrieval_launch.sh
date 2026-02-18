
file_path=/public_hw/share/cit_ztyu/cz/Search-R1/indexing_corpus
index_file=$file_path/e5_HNSW64.index
corpus_file=$file_path/wiki-18.jsonl
retriever_name=e5
retriever_path=/public_hw/share/cit_ztyu/cz/Search-R1/models/e5-base-v2

export HF_HOME=/public_hw/share/cit_ztyu/cz/Search-R1/hf_cache
export HUGGINGFACE_HUB_CACHE=/public_hw/share/cit_ztyu/cz/Search-R1/hf_cache
export OMP_NUM_THREADS=32
export CUDA_VISIBLE_DEVICES=0
python search_r1/search/retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path

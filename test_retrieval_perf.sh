#!/bin/bash

echo "======================================"
echo "Performance Testing Script"
echo "======================================"
echo ""

# 测试单个查询
echo "[Test 1] Single query performance:"
time curl -X POST "http://127.0.0.1:8009/retrieve" \
-H "Content-Type: application/json" \
-d '{
    "queries": ["What is the capital of China?"],
    "topk": 3,
    "return_scores": false
}' -s -o /dev/null -w "Response time: %{time_total}s\n"

echo ""
echo "[Test 2] Batch query performance (10 queries):"
time curl -X POST "http://127.0.0.1:8009/retrieve" \
-H "Content-Type: application/json" \
-d '{
    "queries": [
        "What is artificial intelligence?",
        "Who invented the telephone?",
        "Where is Paris located?",
        "What is quantum computing?",
        "Who wrote Romeo and Juliet?",
        "What is photosynthesis?",
        "When was the internet invented?",
        "What is the speed of light?",
        "Who discovered penicillin?",
        "What is machine learning?"
    ],
    "topk": 3,
    "return_scores": false
}' -s -o /dev/null -w "Response time: %{time_total}s\n"

echo ""
echo "[Test 3] Large batch query performance (100 queries):"
# 生成100个查询
python3 - <<'PYTHON'
import requests
import time
import json

queries = [f"What is topic number {i}?" for i in range(100)]

start = time.time()
response = requests.post(
    "http://127.0.0.1:8009/retrieve",
    json={
        "queries": queries,
        "topk": 3,
        "return_scores": False
    }
)
elapsed = time.time() - start

print(f"100 queries completed in: {elapsed:.3f}s")
print(f"Average per query: {elapsed/100*1000:.2f}ms")
print(f"Throughput: {100/elapsed:.2f} queries/sec")
PYTHON

echo ""
echo "[Test 4] Training-scale batch (128 queries):"
python3 - <<'PYTHON'
import requests
import time

queries = [f"Research topic {i}" for i in range(128)]

start = time.time()
response = requests.post(
    "http://127.0.0.1:8009/retrieve",
    json={
        "queries": queries,
        "topk": 3,
        "return_scores": False
    }
)
elapsed = time.time() - start

print(f"128 queries (1 training batch) completed in: {elapsed:.3f}s")
print(f"Average per query: {elapsed/128*1000:.2f}ms")
print(f"Throughput: {128/elapsed:.2f} queries/sec")
PYTHON

echo ""
echo "======================================"
echo "Testing complete!"
echo "======================================"
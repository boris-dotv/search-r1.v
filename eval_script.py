import asyncio
import argparse
import json
import re
import string
import pandas as pd
import aiohttp
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

# === 关键配置: 训练时的 Prompt 模板 ===
# 必须完全一致，模型才会触发搜索
SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
USER_INSTRUCTION = """Answer the given question. You must conduct reasoning inside <think> and </think> first
every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine
 by <search> query </search> and it will return the top searched results between <information> and </information>. You
 can search as many times as your want. If you find no further external knowledge needed, you can directly provide the
 answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>.
Question: """

# === 工具函数: 文本标准化 ===
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

# === 核心: 检索函数 (带连接池复用和重试) ===
async def search_query(session, url, query, topk=3):
    payload = {"queries": [query], "topk": topk, "return_scores": True}
    max_retries = 5
    
    for attempt in range(max_retries):
        try:
            # 使用共享的 session，避免反复建立 TCP 连接
            async with session.post(url, json=payload, timeout=30) as resp:
                if resp.status != 200:
                    await asyncio.sleep(1)
                    continue
                data = await resp.json()
                return data['result'][0]
        except Exception as e:
            if attempt == max_retries - 1:
                return [] # 最终失败返回空
            await asyncio.sleep(1 + attempt)
    return []

# === 核心: 格式化检索结果 (对齐训练代码) ===
def format_observation(retrieval_result):
    if not retrieval_result:
        return ""
    format_reference = ''
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item['document']['contents']
        parts = content.split("\n")
        title = parts[0]
        text = "\n".join(parts[1:])
        format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
    
    # 必须包含 <information> 标签
    return f'\n\n<information>{format_reference.strip()}</information>\n\n'

# === 核心: 单个问题处理逻辑 ===
async def solve_one_question(row, client, search_session, retriever_url, sem):
    async with sem: # 限制并发数
        try:
            # 1. 数据解析
            try:
                raw_prompt = row['prompt'][0]['content']
                ground_truth_list = row['reward_model']['ground_truth']['target']
                if isinstance(ground_truth_list, str): ground_truth_list = [ground_truth_list]
            except:
                # 兼容不同 Parquet 格式
                raw_prompt = row['question'] if 'question' in row else str(row)
                ground_truth_list = row['answers'] if 'answers' in row else []

            # 2. 构造 Prompt (拼接指令)
            full_content = USER_INSTRUCTION + raw_prompt
            
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": full_content}
            ]
            
            max_turns = 5
            final_answer = ""
            
            # 3. 多轮交互循环
            for turn in range(max_turns):
                response = await client.chat.completions.create(
                    model="default",
                    messages=messages,
                    temperature=0.0, # Greedy Decoding for Eval
                    max_tokens=2048,
                    stop=["</search>", "</answer>"]
                )
                content = response.choices[0].message.content
                
                # SGLang Stop Token 修复逻辑
                # 如果被 stop 截断，手动补全标签
                if not content.endswith(">"):
                    if "<search>" in content and "</search>" not in content:
                        content += "</search>"
                    elif "<answer>" in content and "</answer>" not in content:
                        content += "</answer>"
                
                messages.append({"role": "assistant", "content": content})

                # 正则提取 Action
                match = re.search(r'<(search|answer)>(.*?)</\1>', content, re.DOTALL)
                
                if match:
                    action = match.group(1)
                    inner_content = match.group(2).strip()
                    
                    if action == 'search':
                        # 执行搜索
                        results = await search_query(search_session, retriever_url, inner_content)
                        # 格式化并插入 Observation
                        obs_text = format_observation(results)
                        messages.append({"role": "user", "content": obs_text})
                        continue # 进入下一轮
                        
                    elif action == 'answer':
                        final_answer = inner_content
                        break # 结束循环
                else:
                    # 容错机制：如果没有标签，提示模型重试
                    error_msg = (
                        '\nMy previous action is invalid. '
                        'If I want to search, I should put the query between <search> and </search>. '
                        'If I want to give the final answer, I should put the answer between <answer> and </answer>. '
                        'Let me try again.\n'
                    )
                    messages.append({"role": "user", "content": error_msg})
                    continue

            # 4. 判定结果
            is_correct = False
            if final_answer:
                for gt in ground_truth_list:
                    if exact_match_score(final_answer, gt):
                        is_correct = True
                        break
            
            return is_correct, final_answer, ground_truth_list

        except Exception as e:
            # 捕获单个任务异常，不影响整体
            # print(f"Task Error: {e}")
            return False, "", []

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sglang_url", type=str, required=True)
    parser.add_argument("--retriever_url", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args()

    print(f"Loading data from {args.data_path}")
    df = pd.read_parquet(args.data_path)
    records = df.to_dict('records')
    print(f"Total questions: {len(records)}")

    # 设置 Client 超时
    client = AsyncOpenAI(base_url=f"{args.sglang_url}/v1", api_key="EMPTY", timeout=600.0)
    sem = asyncio.Semaphore(args.concurrency)

    # 关键修改: 创建全局 TCP 连接池
    # limit=0 表示连接池大小无上限（实际由 Semaphore 控制并发请求数）
    # force_close=False 允许长连接复用
    connector = aiohttp.TCPConnector(limit=0, force_close=False)
    
    async with aiohttp.ClientSession(connector=connector) as search_session:
        tasks = [solve_one_question(row, client, search_session, args.retriever_url, sem) for row in records]
        
        correct_count = 0
        total_count = 0
        
        # 使用 tqdm 显示进度
        pbar = tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Evaluating")
        for f in pbar:
            try:
                is_correct, ans, gt = await f
                total_count += 1
                if is_correct:
                    correct_count += 1
                
                # 动态更新进度条后缀
                if total_count % 10 == 0:
                    acc = correct_count / total_count * 100
                    pbar.set_postfix({"Acc": f"{acc:.2f}%", "Done": total_count})
            except Exception:
                pass
                
    final_acc = correct_count / total_count * 100 if total_count > 0 else 0
    print("\n" + "="*50)
    print(f"FINAL RESULTS")
    print(f"Total: {total_count}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {final_acc:.2f}%")
    print("="*50)

if __name__ == "__main__":
    asyncio.run(main())

import asyncio
import json
import os
import re
import traceback
import random

from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Dict, Any, List


class QAJsonFormatter:
    def __init__(self):
        self.passage_pattern = re.compile(r'Passage \d+:\n')

    def split_passages(self, text: str) -> List[str]:
        """Split text into passages based on 'Passage X:' markers."""
        passages = self.passage_pattern.split(text)
        # Remove empty first split if exists
        return [p.strip() for p in passages if p.strip()]

    def extract_title(self, passage: str) -> tuple[str, str]:
        """Extract title and content from a passage."""
        lines = passage.split('\n', 1)
        title = lines[0].strip()
        content = lines[1].strip() if len(lines) > 1 else ""
        return title, content

    def format_passage(self, passage: str, index: int) -> Dict[str, Any]:
        """Format a single passage into a structured dictionary."""
        title, content = self.extract_title(passage)
        return {
            f"passage_{index + 1}": {
                "title": title,
                "content": content
            }
        }

    async def format_qa_json(self, tokenizer, dataset2prompt,
                             maxlen, jsonl_files, dataset_path: str,
                             num_request: int,
                             client_type: str):
        prompts = []
        """Format the entire QA JSON data."""

        # Parse input JSON

        async def process_file(jsonl_file):
            file_prompts = []
            file_path = os.path.join(dataset_path, jsonl_file)
            print(f"正在处理文件: {file_path}")
            prompt_format = dataset2prompt[jsonl_file.split(".")[0]]

            with open(file_path) as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                    else:
                        print(f"文件 {jsonl_file} 为空")
                        continue

                    if client_type == "long":
                        prompt = prompt_format.format(**data)
                        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
                        if len(tokenized_prompt) > maxlen:
                            half = int(maxlen / 2)
                            prompt = tokenizer.decode(tokenized_prompt[:half],
                                                      skip_special_tokens=True) + tokenizer.decode(
                                tokenized_prompt[-half:], skip_special_tokens=True)
                    else:
                        if len(data["conversations"]) >= 2:
                            prompt = data["conversations"][0]["value"]
                        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
                        if len(tokenized_prompt) > (maxlen / 10):
                            half = int(maxlen / 20)
                            prompt = tokenizer.decode(tokenized_prompt[:half],
                                                      skip_special_tokens=True) + tokenizer.decode(
                                tokenized_prompt[-half:], skip_special_tokens=True)

                    file_prompts.append(prompt)
                    if len(file_prompts) > num_request / len(jsonl_files):
                        break
            return file_prompts

        # 创建任务列表
        tasks = []
        for jsonl_file in jsonl_files:
            task = asyncio.create_task(process_file(jsonl_file))
            tasks.append(task)

        # 并发执行所有任务
        file_results = await asyncio.gather(*tasks)

        # 合并所有文件的结果
        for result in file_results:
            prompts.extend(result)
            if len(prompts) > num_request:
                break

        if not prompts:
            print("没有找到有效的对话数据")
            return None

        sampled_ids = [random.randint(0, len(prompts) - 1) for _ in range(num_request)]
        sampled_prompts = [prompts[idx] for idx in sampled_ids]
        return sampled_prompts

async def prepare_benchmark_data(client_type):
    """Prepare and format data for benchmarking"""
    # Load dataset configuration
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))

    # Get data files
    time_data, data_path, jsonl_files = open_jsonl_file(client_type, dataset2prompt)

    # Initialize tokenizer
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained(
        '/Users/myrick/modelHub/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659',
        trust_remote_code=True)

    # Format and filter data
    try:
        formatter = QAJsonFormatter()
        max_samples = 500000 if client_type == 'long' else 100000
        formatted_json = await formatter.format_qa_json(
            tokenizer, dataset2prompt, 5000, jsonl_files, data_path, max_samples, client_type)

        # Filter by length
        filtered_json = []
        str_count = 0
        for item in formatted_json:
            if len(str(item)) < 4000:
                str_count += 1
                filtered_json.append(item)
        print(f'request count: {str_count}')

        return formatted_json, time_data
    except Exception as e:
        print(f"Error: {str(e)}")
        print("详细错误信息:")
        print(traceback.format_exc())
        return None

def open_jsonl_file(client_type, datasets):
    if client_type == "short":
        dataset_path = "sharegpt_gpt4/"
    else:
        dataset_path = "longbench/"

    if not os.path.exists(dataset_path):
        print(f"目录 {dataset_path} 不存在")
        return None

    if not os.path.isdir(dataset_path):
        print(f"{dataset_path} 不是一个目录")
        return None

    jsonl_files = [f for f in os.listdir(dataset_path) if f.endswith('.jsonl')]
    if not jsonl_files:
        print(f"目录 {dataset_path} 中没有找到jsonl文件")
        return None

    filtered_files = []
    for jsonl_file in jsonl_files:
        file_name = jsonl_file.split('.')[0]
        if file_name in datasets:
            filtered_files.append(jsonl_file)
        # else:
        #     print(f"警告: {jsonl_file} 不在预定义的datasets中")

    timedata = load_dataset(
        "/Users/myrick/dataset_hub/datasets--lmsys--chatbot_arena_conversations/snapshots/1b6335d42a1d2c7e34870c905d03ab964f7f2bd8/data/").data[
        'train']['tstamp'].to_pylist()

    for i in reversed(range(len(timedata))):
        if i == 0:
            timedata[i] = 0
            break
        timedata[i] = int(timedata[i] - timedata[i - 1])

    return timedata, dataset_path, filtered_files if filtered_files else None
import os

from config.Config import GLOBAL_CONFIG

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import asyncio
import json
import os
import re
import traceback
import random
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import threading

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

    async def format_qa_json(self, tokenizer, dataset2prompt, maxlen, jsonl_files, dataset_path: str, num_request: int,
                             client_type: str):
        prompts = []
        """Format the entire QA JSON data."""

        # Parse input JSON

        async def process_file(jsonl_file):
            file_prompts = []
            file_path = os.path.join(dataset_path, jsonl_file)
            print(f"正在处理文件: {file_path}")
            # prompt_format = dataset2prompt[jsonl_file.split(".")[0]]

            with open(file_path) as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                    else:
                        print(f"文件 {jsonl_file} 为空")
                        continue

                    if client_type == "long":
                        if len(data["conversations"]) >= 2:
                            prompt = data["conversations"][0]["value"]
                        model_max_len = min(tokenizer.model_max_length, 3000)  # 防止意外
                        tokenized_prompt = tokenizer(prompt, truncation=True, max_length=model_max_len - 256,
                                              return_tensors="pt").input_ids[0]
                        if len(tokenized_prompt) > (maxlen):
                            continue
                        else:
                            file_prompts.append(tokenizer.decode(tokenized_prompt, skip_special_tokens=True))
                    else:
                        if len(data["conversations"]) >= 2:
                            prompt = data["conversations"][0]["value"]
                        model_max_len = min(tokenizer.model_max_length, 3000)  # 防止意外
                        tokenized_prompt = tokenizer(prompt, truncation=True, max_length=model_max_len - 256,
                                                     return_tensors="pt").input_ids[0]
                        if len(tokenized_prompt) > (maxlen / 4):
                            continue
                        else:
                            file_prompts.append(tokenizer.decode(tokenized_prompt, skip_special_tokens=True))

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


async def prepare_benchmark_data(client_type, tokenizer):
    """Prepare and format data for benchmarking"""
    # Load dataset configuration
    dataset2prompt = json.load(open("../config/dataset2prompt.json", "r"))

    # Get data files
    time_data, data_path, jsonl_files = open_jsonl_file(client_type, dataset2prompt)

    # Initialize tokenizer
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Format and filter data
    try:
        formatter = QAJsonFormatter()
        max_samples = 10000
        formatted_json = await formatter.format_qa_json(
            tokenizer, dataset2prompt, GLOBAL_CONFIG.get('prompt_max_len', 256), jsonl_files, data_path, max_samples,
            client_type)

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
    if client_type == "short" or client_type == "long":
        dataset_path = "../sharegpt_gpt4/"
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

    return load_time_data(), dataset_path, filtered_files if filtered_files else None

def validate_timestamps(timestamps):
    """验证时间戳数据的有效性"""
    if not timestamps:
        raise ValueError("Empty timestamp list")

    # 确保时间戳是单调递增的
    for i in range(1, len(timestamps)):
        if timestamps[i] < timestamps[i - 1]:
            raise ValueError(f"Non-monotonic timestamps at index {i}")

    # 检查异常大的时间间隔
    max_reasonable_interval = 3600  # 假设正常间隔不超过1小时
    for i in range(1, len(timestamps)):
        interval = timestamps[i] - timestamps[i - 1]
        if interval > max_reasonable_interval:
            print(f"Warning: Large time interval ({interval}s) detected at index {i}")

def load_time_data(target_qps=None):
    """加载并处理时间数据

    Args:
        target_qps: 可选的目标QPS，用于规范化时间间隔
    """
    try:
        timedata = load_dataset(
            "/Users/myrick/dataset_hub/datasets--lmsys--chatbot_arena_conversations/snapshots/1b6335d42a1d2c7e34870c905d03ab964f7f2bd8/data/"
        ).data['train']['tstamp'].to_pylist()

        return process_timestamps(timedata, target_qps)

    except Exception as e:
        print(f"Error loading time data: {e}")
        return [1.0]  # 返回默认间隔


def process_timestamps(timestamps, target_qps=1):
    """处理时间戳数据，计算并可选地规范化时间间隔

    Args:
        timestamps: 原始时间戳列表
        target_qps: 目标QPS，如果提供则将间隔调整为匹配该QPS
    """
    try:
        validate_timestamps(timestamps)

        # 计算时间间隔
        intervals = [0.0]  # 第一个间隔为0
        for i in range(1, len(timestamps)):
            interval = float(timestamps[i] - timestamps[i - 1])
            intervals.append(interval)

        # 如果指定了目标QPS，调整间隔
        if target_qps is not None and target_qps > 0:
            target_interval = 1.0 / target_qps
            adjusted_intervals = [
                min(interval, target_interval * 2)  # 限制最大间隔
                for interval in intervals
            ]
            return adjusted_intervals

        return intervals

    except Exception as e:
        print(f"Error processing timestamps: {e}")
        return [1.0]  # 返回默认间隔

@lru_cache(maxsize=1024)
def _cached_encode(text, prefix_len, tokenizer):
    """缓存编码结果以避免重复计算"""
    # 只取前缀部分进行编码，避免处理整个长文本
    prefix = text[:prefix_len * 4]  # 估算每个token约4个字符
    return tuple(tokenizer.encode(prefix, add_special_tokens=False)[:prefix_len])


def _get_prefix_key(item, prefix_len, tokenizer):
    """获取排序键"""
    return _cached_encode(item, prefix_len, tokenizer)


def make_prefix_list(data, tokenizer, prefix_len=50, parallel=True):
    """
    对数据列表按token前缀进行排序
    
    Args:
        data: 要排序的数据列表
        tokenizer: 分词器
        prefix_len: 用于排序的前缀长度
        parallel: 是否使用并行处理
    
    Returns:
        排序后的列表
    """
    if not data:
        return data

    if parallel and len(data) > 1000:  # 只在数据量较大时使用并行
        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=min(32, len(data) // 100)) as executor:
            # 创建(item, key)对的列表
            items_with_keys = list(executor.map(
                lambda x: (x, _get_prefix_key(x, prefix_len, tokenizer)),
                data
            ))

        # 按key排序后提取item
        return [item for item, _ in sorted(items_with_keys, key=lambda x: x[1])]
    else:
        # 数据量小时直接排序
        return sorted(data, key=lambda x: _get_prefix_key(x, prefix_len, tokenizer))


if __name__ == "__main__":
    # Create prompt_hub directory if it doesn't exist
    if not os.path.exists('prompt_hub'):
        os.makedirs('prompt_hub')

    tokenizer = AutoTokenizer.from_pretrained(
        "/Users/myrick/modelHub/hub/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
        trust_remote_code=True)

    # Save short context prompts
    short_prompts_path = '../prompt_hub/short_prompts.json'
    short_formatted_json, prompt_time_data = asyncio.run(prepare_benchmark_data('short', tokenizer))
    with open(short_prompts_path, 'w', encoding='utf-8') as f:
        json.dump(short_formatted_json, f, indent=2, ensure_ascii=False)
        print(f"Short prompts saved to {short_prompts_path}")

    # Save long context prompts
    long_prompts_path = '../prompt_hub/long_prompts.json'
    long_formatted_json, _ = asyncio.run(prepare_benchmark_data('long', tokenizer))
    with open(long_prompts_path, 'w', encoding='utf-8') as f:
        json.dump(long_formatted_json, f, indent=2, ensure_ascii=False)
        print(f"Long prompts saved to {long_prompts_path}")

import asyncio
import json
import os
import random
import socket
import threading
import time

import numpy as np
import pandas as pd
from typing import Dict, Any, List
import re
import subprocess
from asyncio import log
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM
import torch
import requests
import os
from datasets import load_dataset
from sshtunnel import SSHTunnelForwarder


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


def monitor_tunnel(server):
    """Monitor tunnel connection and restart if needed"""
    while True:
        if not server.is_active:
            try:
                print(f"Tunnel to {server.ssh_host} disconnected, attempting to restart...")
                server.restart()
                print(f"Tunnel restarted successfully")
            except Exception as e:
                print(f"Error restarting tunnel: {e}")
        time.sleep(1)  # Check every 1 seconds


def create_ssh_tunnel(host, remote_port, local_port, username, password, ssh_port):
    tunnel_servers = None
    try:
        tunnel_servers = SSHTunnelForwarder(
            host,
            ssh_username=username,
            ssh_password=password,
            ssh_port=ssh_port,  # 指定 SSH 登录端口
            remote_bind_address=('localhost', remote_port),
            local_bind_address=('localhost', local_port),
            set_keepalive=20
        )
        tunnel_servers.start()
        print(f"SSH tunnel established on local port {local_port}")

        # Start monitoring in separate thread
        monitor_thread = threading.Thread(target=monitor_tunnel, args=(tunnel_servers,))
        monitor_thread.daemon = True  # Thread will terminate when main program exits
        monitor_thread.start()

    except Exception as e:
        print(f"Error creating SSH tunnel: {e}")

    return tunnel_servers


def stop_tunnel(server):
    """关闭 SSH 隧道"""
    if server.is_active:
        server.stop()
        print("Tunnel to:" + server.ssh_host + "stopped.")


def some_endpoint_test(base_url):
    """Test the vLLM service by sending a small inference request."""
    url = f"{base_url}/health"
    payload = {
        "prompt": "Test prompt for vLLM",
        "max_tokens": 5
    }
    try:
        response = requests.get(url, timeout=10, json=payload)
        if response.status_code == 200:
            # 健康检查成功，解析返回的 JSON 数据
            return {"status": "success", "data": response.json()}
        else:
            # 健康检查失败，记录错误
            return {
                "status": "failure",
                "error": f"HTTP {response.status_code}",
                "body": response.text
            }
    except requests.exceptions.RequestException as e:
        # 网络或请求异常
        return {"status": "error", "exception": str(e)}


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
        else:
            print(f"警告: {jsonl_file} 不在预定义的datasets中")

    timedata = load_dataset(
        "/Users/myrick/dataset_hub/datasets--lmsys--chatbot_arena_conversations/snapshots/1b6335d42a1d2c7e34870c905d03ab964f7f2bd8/data/").data[
        'train']['tstamp'].to_pylist()

    for i in reversed(range(len(timedata))):
        if i == 0:
            timedata[i] = 0
            break
        timedata[i] = int(timedata[i] - timedata[i - 1])

    return timedata, dataset_path, filtered_files if filtered_files else None


def sample_sharegpt_requests(
        dataset_path: str,
        num_requests: int,
        # tokenizer,
        # max_seqlen:int,
):
    # Load the dataset.
    prompts = []
    # prompt_lens = []
    # response_lens = []
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

    for jsonl_file in jsonl_files:
        file_path = os.path.join(dataset_path, jsonl_file)
        print(f"正在处理文件: {file_path}")

        with open(file_path) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                else:
                    print(f"文件 {jsonl_file} 为空")
                    continue

                if len(data["conversations"]) >= 2:
                    prompt = data["conversations"][0]["value"]
                    prompts.append(prompt)
                    # res = data["conversations"][1]["value"]
                    # prompt_token_ids = tokenizer(prompt).input_ids
                    # completion_token_ids = tokenizer(res).input_ids
                    # if len(prompt_token_ids) + len(completion_token_ids) < max_seqlen and \
                    #     len(prompt_token_ids) > 0 and len(completion_token_ids) > 0:
                    #     prompts.append(prompt)
                    #     prompt_lens.append(len(prompt_token_ids))
                    #     response_lens.append(len(completion_token_ids))

                if len(prompts) > num_requests / len(jsonl_files):
                    break

        if len(prompts) > num_requests:
            break

    if not prompts:
        print("没有找到有效的对话数据")
        return None

    sampled_ids = [random.randint(0, len(prompts) - 1) for _ in range(num_requests)]
    sampled_prompts = [prompts[idx] for idx in sampled_ids]
    # sampled_prompt_lens = [prompt_lens[idx] for idx in sampled_ids]
    # sampled_response_lens = [response_lens[idx] for idx in sampled_ids]
    # print(f"max len:{max(a+b for a,b in zip(prompt_lens, response_lens))}")
    return sampled_prompts


def check_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return False  # 端口未被占用
        except socket.error:
            return True  # 端口已被占用


def setup_vllm_servers(vllm_urls, local_port, remote_port):
    if len(vllm_urls) != len(local_port) or len(vllm_urls) != len(remote_port):
        raise ValueError("Number of vLLM URLs must match number of local ports and remote ports")
    """
    Setup vLLM servers by checking URLs, ports and establishing SSH tunnels.
    Returns list of active SSH tunnels.
    """
    # Check number of vLLM URLs

    if len(vllm_urls) > 2:
        print("Warning: More than 2 vLLM URLs provided. Only first 2 will be used.")
        vllm_urls = vllm_urls[:2]
    elif len(vllm_urls) < 1:
        raise ValueError("At least one vLLM URL must be provided")

    servers = []
    # Check ports and establish SSH tunnels
    for i, url in enumerate(vllm_urls):
        # Extract host from URL
        host = url.split("://")[1].split(":")[0]
        port = int(url.split(":")[-1].split("/")[0])

        # Check if port is in use
        if check_port_in_use(port):
            print(f"Port {port} is in use. Attempting to kill existing process...")
            os.system(f"lsof -ti:{port} | xargs kill -9")

        # Start SSH tunnel using command line
        # Use environment variable or config file to store password
        ssh_password = os.environ.get('SSH_PASSWORD', '')  # Get password from environment variable
        if not ssh_password:
            # Fallback to reading from config file if env var not set
            try:
                with open('config/ssh_config.json') as f:
                    ssh_password = json.load(f).get('password', '')
            except:
                raise ValueError("SSH password not found in environment or config file")
        server = create_ssh_tunnel(host, port, local_port[i], 'hjh', ssh_password, remote_port[i])
        print(f"Established SSH tunnel for {url}")
        servers.append(server)
    return servers


def get_target_time(request_count, rate_lambda, global_start_time, distribution, use_time_data, time_data):
    if use_time_data:
        target_time = time_data[request_count]
    else:
        # 计算这个请求应该在什么时间点发送（基于全局开始时间）
        if distribution == "poisson":
            # 泊松分布：请求之间的间隔遵循指数分布
            intervals = [float(np.random.exponential(1 / rate_lambda)) for _ in range(request_count + 1)]
            target_time = global_start_time + sum(intervals)
        elif distribution == "normal":
            # 正态分布：请求均匀分布，但有小的随机波动
            target_time = global_start_time + (request_count / rate_lambda) + float(np.random.normal(0, 0.01))
        else:
            # 均匀分布：请求基本均匀，但有一定范围的随机性
            jitter = np.random.uniform(-0.1, 0.1) / rate_lambda
            target_time = global_start_time + (request_count / rate_lambda) + jitter

    return target_time


def save_results(f_result, s_result, RESULTS_FILE):
    """将公平性结果追加写入 JSON 文件"""
    # 获取当前时间并格式化为24小时制，精确到分钟
    current_time = time.strftime("%H:%M", time.localtime())
    
    new_entry = {
        "f_result": f_result, 
        "s_result": s_result,
        "time": current_time
    }

    # 读取现有数据
    with open(RESULTS_FILE, "r") as f:
        data = json.load(f)

    # 追加新数据
    data.append(new_entry)

    # **写回文件**
    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f, indent=4)

    print(f"结果已写入 {RESULTS_FILE}: {new_entry}")
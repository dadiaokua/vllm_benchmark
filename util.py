import json
import os
import random
from asyncio import log

import requests
from sshtunnel import SSHTunnelForwarder


def start_tunnel():
    """启动 SSH 隧道"""
    server = SSHTunnelForwarder(
        ssh_address_or_host=('222.201.144.120', 10085),  # SSH服务器地址和端口
        ssh_username='hjh',  # SSH登录用户名
        ssh_password='zsygmm1A',  # SSH密码
        remote_bind_address=('http://0.0.0.0:', 8000),  # 远端API地址和端口
        local_bind_address=('127.0.0.1', 8080)  # 本地映射端口
    )
    server.start()
    print(f"Tunnel started: {server.is_active}, Local bind port: {server.local_bind_port}")
    return server


def stop_tunnel(server):
    """关闭 SSH 隧道"""
    if server.is_active:
        server.stop()
        print("Tunnel stopped.")


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
    if os.path.exists(dataset_path):
        with open(dataset_path, 'r') as f:
            print(f"{dataset_path} 文件已成功打开")
    else:
        print(f"文件 {dataset_path} 不存在")
        return None
    with open(dataset_path) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
            else:
                print("Dataset file is empty")
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
            if len(prompts) > num_requests:
                break
    sampled_ids = [random.randint(0, len(prompts) - 1) for _ in range(num_requests)]
    sampled_prompts = [prompts[idx] for idx in sampled_ids]
    # sampled_prompt_lens = [prompt_lens[idx] for idx in sampled_ids]
    # sampled_response_lens = [response_lens[idx] for idx in sampled_ids]
    # print(f"max len:{max(a+b for a,b in zip(prompt_lens, response_lens))}")
    return sampled_prompts

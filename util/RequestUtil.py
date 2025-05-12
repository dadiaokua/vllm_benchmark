import asyncio
import time

import numpy as np
import logging

import random

from config.Config import GLOBAL_CONFIG


async def process_stream(stream):
    first_token_time = None
    total_tokens = 0
    async for chunk in stream:
        if first_token_time is None:
            first_token_time = time.time()
        if chunk.choices[0].delta.content:
            total_tokens += 1
        if chunk.choices[0].finish_reason is not None:
            break
    return first_token_time, total_tokens


async def make_request(client, output_tokens, request_timeout, request, tokenizer, latency_slo):
    start_time = time.time()

    try:
        logging.getLogger("openai").setLevel(logging.ERROR)
        # 使用log_request=False参数来禁止在日志中打印请求内容
        stream = await client.chat.completions.create(
            model="llama_8b",
            messages=[{"role": "user", "content": request}],
            max_tokens=output_tokens,
            stream=True
        )
        first_token_time, total_tokens = await asyncio.wait_for(process_stream(stream), timeout=request_timeout)
        end_time = time.time()
        elapsed_time = end_time - start_time
        ttft = first_token_time - start_time if first_token_time else None
        input_token = tokenizer(request, truncation=False, return_tensors="pt").input_ids[0]
        tokens_per_second = total_tokens / elapsed_time if elapsed_time > 0 else 0
        return total_tokens, elapsed_time, tokens_per_second, ttft, len(
            input_token), 1 if elapsed_time <= latency_slo else 0

    except asyncio.TimeoutError:
        logging.warning(f"Request timed out after {request_timeout} seconds")
        return None
    except Exception as e:
        logging.error(f"Error during request: {str(e)}")
        return None


async def worker(selected_clients, semaphore, results, output_tokens, client_index, tokenizer, request_timeout,
                 round_time, rate_lambda, distribution, sample_content, config_round, worker_id, time_data,
                 use_time_data, latency_slo, time_ratio):
    """每个task发送单个请求，使用get_target_time控制间隔"""
    global_start_time = time.time()
    request_count = 0
    drift_time = 0
    tasks = []

    while time.time() - global_start_time < round_time:
        # 计算当前请求的目标时间
        target_time = get_target_time(
            request_count=request_count,
            rate_lambda=rate_lambda,
            global_start_time=global_start_time,
            distribution=distribution,
            use_time_data=use_time_data,
            time_data=time_data,
            time_ratio=time_ratio,
            window_length=round_time
        )

        # 如果目标时间已经过去，直接发送请求
        if target_time <= time.time():
            drift_time = target_time - time.time()
            # while True:
            #     request = random.choice(sample_content)
            #     input_len = len(tokenizer(request, truncation=False, return_tensors="pt").input_ids[0])
            #     if input_len <= 3000:
            #         break
            request = random.choice(sample_content)
            selected_client = selected_clients[worker_id % len(selected_clients)]

            task = asyncio.create_task(
                process_request(
                    selected_client, output_tokens, request_timeout, [request],  # 注意这里改为单个请求的列表
                    worker_id, tokenizer, results,
                    client_index, semaphore, config_round, latency_slo
                )
            )
            tasks.append(task)
            request_count += 1
        else:
            # 等待到目标时间
            await asyncio.sleep(target_time - time.time())

    # 等待所有任务完成
    if tasks:
        await asyncio.gather(*tasks)

    return request_count, drift_time


async def process_request(client, output_tokens, request_timeout, batch_requests, worker_id, tokenizer,
                          results, client_index, semaphore, config_round, latency_slo):
    async with semaphore:
        request_count = len(batch_requests)
        request_type = "requests" if request_count > 1 else "request"
        for _, request in enumerate(batch_requests):
            try:
                result = await make_request(client, output_tokens, request_timeout, request, tokenizer, latency_slo)
                if result:
                    results.append(result)
            except Exception as e:
                logging.error(
                    f"Worker {worker_id} {config_round + 1} round with {request_count} {request_type} for client {client_index} raised an exception: {e}")


def calculate_raw_request_time(request_count, rate_lambda, global_start_time, distribution):
    """计算基础请求时间，考虑 QPS 要求"""
    if rate_lambda <= 0:
        rate_lambda = 0.001

    # 基础时间间隔
    base_interval = 1 / rate_lambda

    if distribution.lower() == "poisson":
        # 泊松分布：使用指数分布生成间隔
        interval = float(np.random.exponential(base_interval))
    elif distribution.lower() == "normal":
        # 正态分布：在基础间隔上添加小的随机波动
        interval = base_interval + float(np.random.normal(0, base_interval * 0.1))
    else:
        # 均匀分布：在基础间隔上添加均匀随机波动
        interval = base_interval + float(np.random.uniform(-base_interval * 0.1, base_interval * 0.1))

    # 直接返回下一个请求的时间点
    return global_start_time + (request_count * interval)


def get_target_time(request_count, rate_lambda, global_start_time, distribution, use_time_data, time_data, time_ratio,
                    window_length):
    """计算目标请求时间，只考虑 time_ratio 对间隔的影响"""
    if use_time_data:
        return time_data[request_count]

    raw_time = calculate_raw_request_time(request_count, rate_lambda, global_start_time, distribution)
    time_since_start = raw_time - global_start_time

    # 使用非线性映射来避免在窗口末尾堆积请求
    if time_ratio > 1 and time_since_start <= window_length:
        # 使用sigmoid类函数进行平滑映射
        progress = time_since_start / window_length
        # 调整后的进度，保持开始和结束点不变，但中间部分根据time_ratio拉伸
        adjusted_progress = progress ** (1 / time_ratio)
        adjusted_time_since_start = adjusted_progress * window_length
    else:
        # time_ratio <= 1的情况，直接线性缩放
        adjusted_time_since_start = time_since_start * time_ratio

    # 确保调整后的时间不会超出原始窗口
    if adjusted_time_since_start > window_length >= time_since_start:
        adjusted_time_since_start = window_length

    return global_start_time + adjusted_time_since_start

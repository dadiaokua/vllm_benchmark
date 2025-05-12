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


async def make_request(client, output_tokens, request_timeout, requests, tokenizer, latency_slo):
    start_time = time.time()

    try:
        logging.getLogger("openai").setLevel(logging.ERROR)
        # 使用log_request=False参数来禁止在日志中打印请求内容
        stream = await client.chat.completions.create(
            model="llama_8b",
            messages=[{"role": "user", "content": req} for req in requests],
            max_tokens=output_tokens,
            stream=True
        )
        first_token_time, total_tokens = await asyncio.wait_for(process_stream(stream), timeout=request_timeout)
        end_time = time.time()
        elapsed_time = end_time - start_time
        ttft = first_token_time - start_time if first_token_time else None
        input_token = tokenizer(requests, truncation=False, return_tensors="pt").input_ids[0]
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
                 use_time_data, latency_slo, active_ratio, time_ratio):
    """考虑active_ratio和time_ratio的批量发送逻辑"""
    global_start_time = time.time()
    request_count = 0
    drift_time = 0
    tasks = []  # 改为列表存储所有任务

    # 调整每秒请求数
    requests_per_second = max(1, rate_lambda)

    # 计算活跃窗口
    window_size = min(GLOBAL_CONFIG['max_granularity'], round_time)
    active_duration = window_size * active_ratio

    next_batch_time = time.time()  # 下一批次的目标时间

    while time.time() - global_start_time < round_time:
        # 发送这一秒的请求批次
        batch_requests = []
        for i in range(requests_per_second):
            request = sample_content[request_count % len(sample_content)]
            batch_requests.append(request)

        selected_client = selected_clients[worker_id % len(selected_clients)]
        if time.time() % window_size < active_duration:
            task = asyncio.create_task(
                process_request(
                    selected_client, output_tokens, request_timeout, batch_requests,
                    worker_id, tokenizer, results,
                    client_index, semaphore, config_round, latency_slo
                )
            )
            tasks.append(task)  # 将任务添加到列表中
        else:
            await asyncio.sleep(window_size * (1 - active_ratio))
            task = asyncio.create_task(
                process_request(
                    selected_client, output_tokens, request_timeout, batch_requests,
                    worker_id, tokenizer, results,
                    client_index, semaphore, config_round, latency_slo
                )
            )
            tasks.append(task)  # 将任务添加到列表中

        request_count += requests_per_second

        # 更精确的时间控制
        next_batch_time += 1.0  # 下一批次应该在1秒后
        wait_time = next_batch_time - time.time()
        if wait_time > 0:
            await asyncio.sleep(wait_time)

    # 等待所有任务完成
    if tasks:
        await asyncio.gather(*tasks)

    return request_count, 0


async def process_request(client, output_tokens, request_timeout, batch_requests, worker_id, tokenizer,
                          results, client_index, semaphore, config_round, latency_slo):
    async with semaphore:
        request_count = len(batch_requests)
        request_type = "requests" if request_count > 1 else "request"
        for request_index, request in enumerate(batch_requests):
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


def get_target_time(request_count, rate_lambda, global_start_time, distribution, use_time_data, time_data, active_ratio,
                    window_length, time_ratio):
    """计算目标请求时间，考虑 time_ratio 对间隔的影响"""
    if use_time_data:
        return time_data[request_count]
    else:
        raw_time = calculate_raw_request_time(request_count, rate_lambda, global_start_time, distribution)

    # 应用time_ratio调整整体发送时间
    time_since_start = raw_time - global_start_time

    # 使用非线性映射来避免在窗口末尾堆积请求
    # 当time_ratio > 1时，使用一个平滑的函数来分散请求
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

    # 控制发送时间段 - 更频繁的活跃/非活跃切换
    time_since_start = adjusted_time_since_start

    # 使用更小的窗口长度，以实现更频繁的切换
    small_window_length = min(GLOBAL_CONFIG['max_granularity'], window_length)  # 默认使用10秒的小窗口，除非原窗口更小

    # 计算在小窗口内的位置
    small_window_position = time_since_start % small_window_length
    active_duration = small_window_length * active_ratio

    if small_window_position < active_duration:
        # 当前时间在可发时间段内，直接返回
        return raw_time
    else:
        # 在不可发时间段 - 推迟到下一个小窗口的开始
        wait_time = small_window_length - small_window_position

        # 可选：添加一些随机性，避免所有请求都在窗口开始时堆积
        random_offset = np.random.uniform(0, active_duration)  # 小的随机偏移

        adjusted_time = raw_time + wait_time + random_offset
        return adjusted_time

import asyncio
import time
from datetime import datetime

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


def calculate_all_request_times(rate_lambda, round_time, distribution, time_ratio):
    """预先计算所有请求的时间点"""
    if rate_lambda <= 0:
        rate_lambda = 0.001

    # 基础时间间隔
    base_interval = 1 / rate_lambda

    # 估算总请求数
    estimated_requests = int(round_time * rate_lambda)

    # 生成所有请求的时间点
    request_times = []
    global_start_time = time.time()  # 使用当前时间作为全局开始时间

    # 先生成基础时间点（相对于开始时间的偏移）
    base_times = []
    current_offset = 0
    for i in range(estimated_requests):
        if distribution.lower() == "poisson":
            interval = float(np.random.exponential(base_interval))
        elif distribution.lower() == "normal":
            interval = base_interval + float(np.random.normal(0, base_interval * 0.1))
        else:
            interval = base_interval + float(np.random.uniform(-base_interval * 0.1, base_interval * 0.1))

        current_offset += interval
        if current_offset > round_time:  # 确保不超出round_time
            break
        base_times.append(current_offset)

    # 应用非线性映射
    for base_offset in base_times:
        if time_ratio > 1 and base_offset <= round_time:
            # 使用sigmoid类函数进行平滑映射
            progress = base_offset / round_time
            # 调整后的进度，保持开始和结束点不变，但中间部分根据time_ratio拉伸
            adjusted_progress = progress ** (1 / time_ratio)
            adjusted_offset = adjusted_progress * round_time
        else:
            # time_ratio <= 1的情况，直接线性缩放
            adjusted_offset = base_offset * time_ratio

        # 确保调整后的时间不会超出原始窗口
        if adjusted_offset > round_time:
            adjusted_offset = round_time

        # 将偏移转换为绝对时间
        request_time = global_start_time + adjusted_offset
        request_times.append(request_time)

    return request_times


async def worker(selected_clients, semaphore, results, output_tokens, client_index, tokenizer, request_timeout,
                 round_time, rate_lambda, distribution, sample_content, config_round, worker_id, time_data,
                 use_time_data, latency_slo, time_ratio):
    """每个task发送单个请求，使用预先计算的时间点控制间隔"""
    global_start_time = time.time()
    request_count = 0
    drift_time = 0
    completed = 0
    tasks = []
    task_status = {}

    # 预先计算所有请求的时间点
    request_times = calculate_all_request_times(rate_lambda, round_time, distribution, time_ratio)

    for target_time in request_times:
        if time.time() - global_start_time >= round_time:
            break
        current_time = time.time()
        if target_time <= current_time:
            # 如果目标时间已过，直接发送请求
            drift_time = current_time - target_time
        else:
            # 如果还没到目标时间，先sleep
            sleep_time = target_time - current_time
            if sleep_time > 0:
                sleep_start = time.time()
                await asyncio.sleep(sleep_time)
                if sleep_time > 2:
                    sleep_end = time.time()
                    print(f"[Worker {worker_id}] target_time: {target_time:.6f}, "
                          f"current_time: {datetime.fromtimestamp(current_time).strftime('%H:%M:%S.%f')}, "
                          f"sleep_time: {datetime.fromtimestamp(sleep_time).strftime('%H:%M:%S.%f')}, "
                          f"actual_sleep: {sleep_end - sleep_start:.6f}")
            else:
                print(f"[Worker {worker_id}] Warning: Negative sleep time detected: {sleep_time:.6f} seconds")
                continue

        # 发送请求（不管是否需要sleep，都会执行到这里）
        request = random.choice(sample_content)
        selected_client = selected_clients[worker_id % len(selected_clients)]
        task = asyncio.create_task(
            process_request(
                selected_client, output_tokens, request_timeout, request,
                worker_id, tokenizer, results,
                client_index, semaphore, config_round, latency_slo
            )
        )
        task_status[task] = {"start_time": time.time(), "status": "running"}
        task.add_done_callback(lambda t: task_status.update({t: {"status": "completed", "end_time": time.time()}}))
        tasks.append(task)
        request_count += 1

    remaining_time = round_time - (time.time() - global_start_time)
    if remaining_time > 0:
        await asyncio.sleep(remaining_time)
        print(f"[Worker {worker_id}] Warning: Not enough requests to fill the round time. Sleeping for {remaining_time:.2f} seconds")

    # 等待所有任务完成
    if tasks:
        completed = sum(1 for status in task_status.values() if status["status"] == "completed")
        print(f"Total tasks: {request_count}, Completed: {completed}")
        print(f"Task completion rate: {completed / len(tasks) * 100:.2f}%")

    return completed, drift_time, request_count


async def process_request(client, output_tokens, request_timeout, request, worker_id, tokenizer,
                          results, client_index, semaphore, config_round, latency_slo):
    async with semaphore:
        try:
            result = await make_request(client, output_tokens, request_timeout, request, tokenizer, latency_slo)
            if result:
                results.append(result)
        except Exception as e:
            logging.error(
                f"Worker {worker_id} {config_round + 1} round for client {client_index} raised an exception: {e}")

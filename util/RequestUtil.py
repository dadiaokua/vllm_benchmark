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


async def make_request(client, experiment, request):
    start_time = time.time()
    try:
        # 使用log_request=False参数来禁止在日志中打印请求内容
        stream = await client.chat.completions.create(
            model=GLOBAL_CONFIG['request_model_name'],
            messages=[{"role": "user", "content": request}],
            max_tokens=experiment.output_tokens,
            stream=True
        )
        first_token_time, total_tokens = await asyncio.wait_for(process_stream(stream), timeout=experiment.request_timeout)
        end_time = time.time()
        elapsed_time = end_time - start_time
        ttft = first_token_time - start_time if first_token_time else None
        input_token = experiment.tokenizer(request, truncation=False, return_tensors="pt").input_ids[0]
        tokens_per_second = total_tokens / elapsed_time if elapsed_time > 0 else 0
        return total_tokens, elapsed_time, tokens_per_second, ttft, len(
            input_token), 1 if elapsed_time <= experiment.latency_slo else 0

    except asyncio.TimeoutError:
        experiment.logger.warning(f"Request timed out after {experiment.request_timeout} seconds")
        return None
    except Exception as e:
        experiment.logger.error(f"Error during request: {str(e)}")
        return None


def calculate_all_request_times(rate_lambda, round_time, distribution, time_ratio):
    """
    预先计算所有请求的时间点
    
    Args:
        rate_lambda: 每分钟发送的请求数量
        round_time: 测试轮次时间（秒）
        distribution: 分布类型
        time_ratio: 时间比例
    
    Returns:
        list: 请求时间点列表
    """
    # 将每分钟请求数转换为每秒请求数
    rate_per_second = rate_lambda / 60.0
    
    if rate_per_second <= 0:
        rate_per_second = 0.001

    # 基础时间间隔
    base_interval = 1 / rate_per_second

    # 估算总请求数
    estimated_requests = int(round_time * rate_per_second)

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


async def worker(experiment, selected_clients, semaphore, results, worker_id, worker_json, qpm_per_worker):
    """每个task发送单个请求，使用预先计算的时间点控制间隔"""
    assert worker_json is not None, "sample_content is None!"
    assert isinstance(worker_json, list), f"sample_content is not a list! type={type(worker_json)}"
    assert len(worker_json) > 0, "sample_content is empty!"
    global_start_time = time.time()
    request_count = 0
    drift_time = 0
    completed = 0
    tasks = []
    task_status = {}

    # 预先计算所有请求的时间点
    request_times = calculate_all_request_times(qpm_per_worker, experiment.round_time, experiment.distribution,
                                                experiment.time_ratio)

    for target_time in request_times:
        if time.time() - global_start_time >= experiment.round_time:
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
                    experiment.logger.info(f"[Worker {worker_id}] target_time: {target_time:.6f}, "
                                           f"current_time: {datetime.fromtimestamp(current_time).strftime('%H:%M:%S.%f')}, "
                                           f"sleep_time: {datetime.fromtimestamp(sleep_time).strftime('%H:%M:%S.%f')}, "
                                           f"actual_sleep: {sleep_end - sleep_start:.6f}")
            else:
                experiment.logger.warning(
                    f"[Worker {worker_id}] Warning: Negative sleep time detected: {sleep_time:.6f} seconds")
                continue

        # 发送请求（不管是否需要sleep，都会执行到这里）
        request = random.choice(worker_json)
        selected_client = selected_clients[worker_id % len(selected_clients)]
        task = asyncio.create_task(
            process_request(selected_client, experiment, request, worker_id, results, semaphore)
        )
        task_status[task] = {"start_time": time.time(), "status": "running"}
        task.add_done_callback(lambda t: task_status.update({t: {"status": "completed", "end_time": time.time()}}))
        tasks.append(task)
        request_count += 1

    elapsed = time.time() - global_start_time
    remaining_time = experiment.round_time - elapsed
    if remaining_time > 3:  # 只在剩余时间大于3秒时才sleep，防止误差
        experiment.logger.warning(
            f"[Worker {worker_id}] Warning: Not enough requests to fill the round time. Sleeping for {remaining_time:.2f} seconds")
        await asyncio.sleep(remaining_time)
    else:
        experiment.logger.info(f"[Worker {worker_id}] Finished all requests, no need to sleep.")

    # 等待所有任务完成
    if tasks:
        completed = sum(1 for status in task_status.values() if status["status"] == "completed")
        experiment.logger.info(f"Total tasks: {request_count}, Completed: {completed}")
        experiment.logger.info(f"Task completion rate: {completed / len(tasks) * 100:.2f}%")

    return completed, drift_time, request_count


async def process_request(client, experiment, request, worker_id, results, semaphore):
    async with semaphore:
        try:
            result = await make_request(client, experiment, request)
            if result:
                results.append(result)
        except Exception as e:
            logging.error(
                f"Worker {worker_id} {experiment.config_round + 1} round for client {experiment.client_index} raised an exception: {e}")

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
            messages=[
                {"role": "user", "content": request}
            ],
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
                 use_time_data, latency_slo, active_ratio):
    # 使用全局时间基准点，而不是相对于上一个请求的时间
    global_start_time = time.time()
    request_count = 0

    while time.time() - global_start_time < round_time:
        target_time = get_target_time(request_count, rate_lambda, global_start_time, distribution, use_time_data,
                                      time_data, active_ratio, round_time)

        # 计算需要等待的时间
        now = time.time()
        wait_time = max(0, target_time - now)

        if wait_time > 0:
            await asyncio.sleep(wait_time)

        # 记录实际请求发送时间
        actual_time = time.time()
        time_str = time.strftime('%H:%M:%S.%f', time.localtime(actual_time))[:-3]
        drift = actual_time - target_time

        task_id = request_count  # 使用请求计数作为task_id

        logging.info(
            f"Client {client_index}, Worker {worker_id}, {config_round + 1} round task {task_id}: Actual={time_str}, "
            f"Target={time.strftime('%H:%M:%S.%f', time.localtime(target_time))[:-3]}, "
            f"Drift={drift:.3f}s, QPS target={rate_lambda}")

        # 随机选择一个请求内容
        request = random.choice(sample_content)

        # 根据task_id轮询选择客户端
        selected_client = selected_clients[task_id % len(selected_clients)]

        # 创建并发送异步请求，但不等待它完成
        await asyncio.create_task(
            process_request(selected_client, output_tokens, request_timeout, request, worker_id, tokenizer,
                            results, task_id, client_index, semaphore, config_round, latency_slo)
        )

        # 增加请求计数
        request_count += 1

    return request_count


async def process_request(client, output_tokens, request_timeout, request, worker_id, tokenizer,
                          results, task_id, client_index, semaphore, config_round, latency_slo):
    async with semaphore:
        logging.info(
            f"Starting worker {worker_id} {config_round + 1} round request {task_id} for client {client_index}")
        try:
            result = await make_request(client, output_tokens, request_timeout, request, tokenizer, latency_slo)
            if result:
                results.append(result)
            else:
                logging.warning(
                    f"Worker {worker_id} {config_round + 1} round Request {task_id} failed for client {client_index}")
        except Exception as e:
            logging.error(
                f"Worker {worker_id} {config_round + 1} round Request {task_id} for client {client_index} raised an exception: {e}")
        logging.info(
            f"Finished worker {worker_id} {config_round + 1} round request {task_id} for client {client_index}")


def get_target_time(request_count, rate_lambda, global_start_time, distribution, use_time_data, time_data, active_ratio,
                    window_length):
    if use_time_data:
        return time_data[request_count]
    else:
        raw_time = calculate_raw_request_time(request_count, rate_lambda, global_start_time, distribution)

        # 控制发送时间段 - 更频繁的活跃/非活跃切换
        time_since_start = raw_time - global_start_time

        # 使用更小的窗口长度，以实现更频繁的切换
        # 例如，如果原来的窗口是100秒，现在我们使用10秒的小窗口
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
            random_offset = np.random.uniform(0, active_duration)  # 小的随机偏移，最多0.1秒

            adjusted_time = raw_time + wait_time + random_offset
            return adjusted_time


def calculate_raw_request_time(request_count, rate_lambda, global_start_time, distribution):
    # 防止除零错误
    if rate_lambda <= 0:
        rate_lambda = 0.001  # 设置一个最小值

    # 计算这个请求应该在什么时间点发送（基于全局开始时间）
    if distribution.lower() == "poisson":
        # 正确实现泊松过程：只需要一个间隔时间
        # 对于第 n 个请求，我们只关心它与第 n-1 个请求之间的间隔
        if request_count == 0:
            # 第一个请求
            interval = float(np.random.exponential(1 / rate_lambda))
            raw_time = global_start_time + interval
        else:
            # 后续请求 - 需要前一个请求的时间
            # 注意：这需要一种方式来获取前一个请求的时间
            # 这里假设我们有一个 previous_request_time 变量
            # 实际实现中，你可能需要传入这个值或维护一个状态
            previous_request_time = global_start_time + (request_count - 1) / rate_lambda  # 简化估计
            interval = float(np.random.exponential(1 / rate_lambda))
            raw_time = previous_request_time + interval
    elif distribution.lower() == "normal":
        # 正态分布：请求均匀分布，但有小的随机波动
        raw_time = global_start_time + (request_count / rate_lambda) + float(np.random.normal(0, 0.01))
    else:
        # 均匀分布：请求基本均匀，但有一定范围的随机性
        jitter = np.random.uniform(-0.1, 0.1) / rate_lambda
        raw_time = global_start_time + (request_count / rate_lambda) + jitter

    return raw_time

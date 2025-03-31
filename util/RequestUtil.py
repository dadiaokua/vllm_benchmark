import asyncio
import time

import numpy as np
import logging

import random


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
        return total_tokens, elapsed_time, tokens_per_second, ttft, len(input_token), 1 if elapsed_time <= latency_slo else 0

    except asyncio.TimeoutError:
        logging.warning(f"Request timed out after {request_timeout} seconds")
        return None
    except Exception as e:
        logging.error(f"Error during request: {str(e)}")
        return None


async def worker(selected_clients, semaphore, results, output_tokens, client_index, tokenizer, request_timeout, round_time,
                 rate_lambda, distribution, sample_content, config_round, worker_id, time_data, use_time_data, latency_slo):
    # 使用全局时间基准点，而不是相对于上一个请求的时间
    global_start_time = time.time()
    request_count = 0

    while time.time() - global_start_time < round_time:
        target_time = get_target_time(request_count, rate_lambda, global_start_time, distribution, use_time_data,
                                      time_data)

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
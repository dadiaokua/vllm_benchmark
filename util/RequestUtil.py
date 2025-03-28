import asyncio
import time

import numpy as np
import logging

import random

from util.MathUtil import calculate_percentile
from util.RequestDistributionUtil import get_target_time


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


async def make_request(client, output_tokens, request_timeout, request, tokenizer):
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
        logging.getLogger("openai").setLevel(logging.INFO)
        return total_tokens, elapsed_time, tokens_per_second, ttft, len(input_token)

    except asyncio.TimeoutError:
        logging.warning(f"Request timed out after {request_timeout} seconds")
        return None
    except Exception as e:
        logging.error(f"Error during request: {str(e)}")
        return None


async def worker(selected_clients, semaphore, results, output_tokens, client_index, tokenizer, request_timeout, round_time,
                 rate_lambda, distribution, sample_content, config_round, worker_id, time_data, use_time_data):
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
                            results, task_id, client_index, semaphore, config_round)
        )

        # 增加请求计数
        request_count += 1

    return request_count


async def process_request(client, output_tokens, request_timeout, request, worker_id, tokenizer,
                          results, task_id, client_index, semaphore, config_round):
    async with semaphore:
        logging.info(
            f"Starting worker {worker_id} {config_round + 1} round request {task_id} for client {client_index}")
        try:
            result = await make_request(client, output_tokens, request_timeout, request, tokenizer)
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


async def run_benchmark(self, concurrency, request_timeout, output_tokens, openAI_clients, distribution, qps, client_index,
                        formatted_json, config_round, tokenizer, time_data, use_time_data, round_time):
    assert openAI_clients is not None, "OpenAI Client must not be None"
    semaphore = asyncio.Semaphore(self.concurrency)
    results = []
    workers = []
    start_time = time.time()
    qps_per_worker = qps / concurrency
    if qps < concurrency:
        qps_per_worker = qps
        concurrency = 1
    for worker_id in range(concurrency):
        worker_task = asyncio.create_task(
            worker(openAI_clients, semaphore, results, output_tokens, client_index, tokenizer, request_timeout, round_time,
                   qps_per_worker, distribution, formatted_json, config_round, worker_id, time_data, use_time_data)
        )
        workers.append(worker_task)

    # Wait for all workers to complete and collect their results
    worker_results = await asyncio.gather(*workers)

    # Sum up all worker request counts
    num_requests = sum(worker_results)

    end_time = time.time()

    # Calculate metrics
    total_elapsed_time = end_time - start_time
    total_tokens = sum(tokens for tokens, _, _, _, _ in results if tokens is not None)
    total_input_tokens = sum(input_token for _, _, _, _, input_token in results if input_token is not None)
    latencies = [elapsed_time for _, elapsed_time, _, _, _ in results if elapsed_time is not None]
    tokens_per_second_list = [tps for _, _, tps, _, _ in results if tps is not None]
    ttft_list = [ttft for _, _, _, ttft, _ in results if ttft is not None]

    successful_requests = len(results)
    requests_per_second = successful_requests / total_elapsed_time if total_elapsed_time > 0 else 0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    avg_tokens_per_second = sum(tokens_per_second_list) / len(tokens_per_second_list) if tokens_per_second_list else 0
    avg_ttft = sum(ttft_list) / len(ttft_list) if ttft_list else 0

    # Calculate percentiles
    percentiles = [50, 95, 99]
    latency_percentiles = [calculate_percentile(latencies, p) for p in percentiles]
    tps_percentiles = [calculate_percentile(tokens_per_second_list, p, reverse=True) for p in percentiles]
    ttft_percentiles = [calculate_percentile(ttft_list, p) for p in percentiles]

    return {
        "time": end_time,
        "qps": qps,
        "total_requests": num_requests,
        "successful_requests": successful_requests,
        "concurrency": concurrency,
        "request_timeout": request_timeout,
        "max_output_tokens": output_tokens,
        "total_time": total_elapsed_time,
        "requests_per_second": requests_per_second,
        "total_output_tokens": total_tokens,
        "total_input_tokens": total_input_tokens,
        "latency": {
            "average": avg_latency,
            "p50": latency_percentiles[0],
            "p95": latency_percentiles[1],
            "p99": latency_percentiles[2]
        },
        "tokens_per_second": {
            "average": avg_tokens_per_second,
            "p50": tps_percentiles[0],
            "p95": tps_percentiles[1],
            "p99": tps_percentiles[2]
        },
        "time_to_first_token": {
            "average": avg_ttft,
            "p50": ttft_percentiles[0],
            "p95": ttft_percentiles[1],
            "p99": ttft_percentiles[2]
        },
        "client_index": client_index,
    }
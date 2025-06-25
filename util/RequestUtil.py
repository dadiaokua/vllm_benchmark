import asyncio
import time
from datetime import datetime
from typing import Any
import uuid

import numpy as np
import logging

import random
import threading
from vllm import SamplingParams
from config.Config import GLOBAL_CONFIG
from util.ThreadSafeUtil import ThreadSafeCounter


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


async def make_request_direct_engine(engine, experiment, request, start_time=None):
    """
    直接使用AsyncLLMEngine处理请求
    
    Args:
        engine: AsyncLLMEngine实例
        experiment: 实验对象
        request: 请求内容
        start_time: 开始时间
        
    Returns:
        tuple: (output_tokens, elapsed_time, tokens_per_second, ttft, input_tokens, slo_met)
    """
    if start_time is None:
        start_time = time.time()

    # 生成唯一的请求ID
    request_id = str(uuid.uuid4())

    try:
        # 注册请求ID到实验的客户端（如果可用）
        if hasattr(experiment, 'client') and hasattr(experiment.client, 'register_request_id'):
            experiment.client.register_request_id(request_id)
        
        # 从配置中获取采样参数
        temperature = GLOBAL_CONFIG.get('sampling_temperature', 0.7)
        top_p = GLOBAL_CONFIG.get('sampling_top_p', 0.9)
        top_k = GLOBAL_CONFIG.get('sampling_top_k', -1)
        repetition_penalty = GLOBAL_CONFIG.get('sampling_repetition_penalty', 1.0)
        
        # 创建采样参数
        sampling_params_dict = {
            "max_tokens": experiment.output_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
        }
        
        # 只有当top_k > 0时才添加top_k参数
        if top_k > 0:
            sampling_params_dict["top_k"] = top_k
            
        sampling_params = SamplingParams(**sampling_params_dict)
        
        # 生成请求
        results = []
        first_token_time = None
        
        # 使用AsyncLLMEngine生成
        async for request_output in engine.generate(request, sampling_params, request_id, priority=experiment.priority):
            if first_token_time is None and len(request_output.outputs) > 0 and len(request_output.outputs[0].token_ids) > 0:
                first_token_time = time.time()
            results.append(request_output)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # 获取最终结果
        if results:
            final_output = results[-1]
            if final_output.outputs:
                output_text = final_output.outputs[0].text
                output_tokens = len(final_output.outputs[0].token_ids)
            else:
                output_tokens = 0
        else:
            output_tokens = 0
            
        ttft = first_token_time - start_time if first_token_time else None
        input_token = experiment.tokenizer(request, truncation=False, return_tensors="pt").input_ids[0]
        tokens_per_second = output_tokens / elapsed_time if elapsed_time > 0 else 0

        # 正常完成时注销请求ID
        if hasattr(experiment, 'client') and hasattr(experiment.client, 'unregister_request_id'):
            experiment.client.unregister_request_id(request_id)

        return output_tokens, elapsed_time, tokens_per_second, ttft, len(
            input_token), 1 if elapsed_time <= experiment.latency_slo else 0

    except asyncio.TimeoutError:
        end_time = time.time()
        # 记录timeout次数
        if hasattr(experiment, 'timeout_count'):
            experiment.timeout_count += 1

        # 超时时也要注销请求ID
        if hasattr(experiment, 'client') and hasattr(experiment.client, 'unregister_request_id'):
            experiment.client.unregister_request_id(request_id)

        experiment.logger.warning(
            f"Client {experiment.client_id} request timed out after {end_time - start_time} seconds (Total timeouts: {experiment.timeout_count})")

        return None
    except Exception as e:
        # 异常时也要注销请求ID
        if hasattr(experiment, 'client') and hasattr(experiment.client, 'unregister_request_id'):
            experiment.client.unregister_request_id(request_id)
        
        experiment.logger.error(f"Error during direct engine request: {str(e)}")
        return None


async def make_request(client, experiment, request, start_time=None):
    """
    发送请求 - 自动检测使用直接引擎还是HTTP客户端
    """
    # 检查是否有直接的vLLM引擎
    if 'vllm_engine' in GLOBAL_CONFIG and GLOBAL_CONFIG['vllm_engine'] is not None:
        # 使用直接引擎API
        return await make_request_direct_engine(GLOBAL_CONFIG['vllm_engine'], experiment, request, start_time)
    else:
        # 使用HTTP客户端（原有方式）
        return await make_request_http_client(client, experiment, request, start_time)


async def make_request_http_client(client, experiment, request, start_time=None):
    """
    使用HTTP客户端处理请求（原有方式）
    """
    if start_time is None:
        start_time = time.time()

    # 生成唯一的请求ID
    request_id = str(uuid.uuid4())

    try:
        # 注册请求ID到实验的客户端（如果可用）
        if hasattr(experiment, 'client') and hasattr(experiment.client, 'register_request_id'):
            experiment.client.register_request_id(request_id)
        
        # 使用log_request=False参数来禁止在日志中打印请求内容
        stream = await client.chat.completions.create(
            model=GLOBAL_CONFIG['request_model_name'],
            messages=[{"role": "user", "content": request}],
            max_tokens=experiment.output_tokens,
            stream=True
            # 注意：移除 extra_headers，因为 OpenAI 客户端可能不支持
            # 请求ID仍然会被跟踪，但不会通过header传递给服务器
        )
        first_token_time, output_tokens = await asyncio.wait_for(process_stream(stream),
                                                                 timeout=experiment.request_timeout)
        end_time = time.time()
        elapsed_time = end_time - start_time
        ttft = first_token_time - start_time if first_token_time else None
        input_token = experiment.tokenizer(request, truncation=False, return_tensors="pt").input_ids[0]
        tokens_per_second = output_tokens / elapsed_time if elapsed_time > 0 else 0

        # 正常完成时注销请求ID
        if hasattr(experiment, 'client') and hasattr(experiment.client, 'unregister_request_id'):
            experiment.client.unregister_request_id(request_id)

        return output_tokens, elapsed_time, tokens_per_second, ttft, len(
            input_token), 1 if elapsed_time <= experiment.latency_slo else 0

    except asyncio.TimeoutError:
        end_time = time.time()
        # 记录timeout次数
        if hasattr(experiment, 'timeout_count'):
            experiment.timeout_count += 1

        # 超时时也要注销请求ID
        if hasattr(experiment, 'client') and hasattr(experiment.client, 'unregister_request_id'):
            experiment.client.unregister_request_id(request_id)

        experiment.logger.warning(
            f"Client {experiment.client_id} request timed out after {end_time - start_time} seconds (Total timeouts: {experiment.timeout_count})")

        return None
    except Exception as e:
        # 异常时也要注销请求ID
        if hasattr(experiment, 'client') and hasattr(experiment.client, 'unregister_request_id'):
            experiment.client.unregister_request_id(request_id)
        
        experiment.logger.error(f"Error during request: {str(e)}")
        return None


async def make_request_via_queue(queue_manager, client_id: str, worker_id: str,
                                 request_content: str, experiment, priority: int = 0) -> Any:
    """通过队列管理器发送请求"""
    try:
        # 提交请求到队列
        request_id = await queue_manager.submit_request(
            client_id=client_id,
            worker_id=worker_id,
            request_content=request_content,
            experiment=experiment,
            priority=priority,
            start_time=time.time()
        )

        # 等待响应
        result = await queue_manager.get_response(client_id, timeout=1000)

        return result

    except Exception as e:
        experiment.logger.error(f"Error making request via queue: {e}")
        return None


def calculate_all_request_times(experiment, qmp_per_worker):
    """
    预先计算所有请求的时间点
    
    Args:
        experiment: 实验对象，包含round_time, distribution, time_ratio等属性
        qmp_per_worker: 每个worker每分钟发送的请求数量
    
    Returns:
        list: 请求时间点列表
    """
    # 从experiment对象中获取参数
    rate_lambda = qmp_per_worker
    round_time = experiment.round_time
    distribution = experiment.distribution
    time_ratio = experiment.time_ratio

    # 预留缓冲时间给最后的请求完成
    buffer_time = round_time * GLOBAL_CONFIG.get('buffer_ratio', 0.5)
    # 确保缓冲时间不超过round_time的50%
    buffer_time = min(buffer_time, round_time * 0.5)
    # 实际可用的发送时间窗口
    effective_round_time = round_time - buffer_time

    experiment.logger.info(
        f"Round time: {round_time}s, Buffer time: {buffer_time}s, Effective time: {effective_round_time}s")

    # 将每分钟请求数转换为每秒请求数
    rate_per_second = rate_lambda / 60.0

    if rate_per_second <= 0:
        rate_per_second = 0.001

    # 基础时间间隔
    base_interval = 1 / rate_per_second

    # 估算总请求数，基于有效时间窗口
    estimated_requests = int(effective_round_time * rate_per_second)
    # 在估算请求数基础上增加10%的随机变化
    random_variation = random.uniform(0.9, 1.1)
    estimated_requests = int(estimated_requests * random_variation)

    # 生成所有请求的时间点
    request_times = []
    global_start_time = time.time()  # 使用当前时间作为全局开始时间

    # 添加一个随机的开始偏移，避免所有client同时开始
    start_offset = random.uniform(0, min(5.0, effective_round_time * 0.1))  # 最多5秒或effective_round_time的10%

    # 先生成基础时间点（相对于开始时间的偏移）
    base_times = []
    current_offset = start_offset  # 从随机偏移开始

    for i in range(estimated_requests):
        # 增加分布的随机性
        if distribution.lower() == "poisson":
            # 泊松分布，但添加更多随机性
            base_rate_variation = random.uniform(0.7, 1.3)  # 基础速率的变化
            adjusted_interval = base_interval * base_rate_variation
            interval = float(np.random.exponential(adjusted_interval))
        elif distribution.lower() == "normal":
            # 正态分布，增加标准差
            std_dev = base_interval * random.uniform(0.2, 0.4)  # 随机标准差
            interval = base_interval + float(np.random.normal(0, std_dev))
            interval = max(0.001, interval)  # 确保间隔为正
        else:
            # 均匀分布，增加变化范围
            variation_range = random.uniform(0.3, 0.7)  # 随机变化范围
            interval = base_interval + float(np.random.uniform(-base_interval * variation_range,
                                                               base_interval * variation_range))
            interval = max(0.001, interval)  # 确保间隔为正

        current_offset += interval
        if current_offset > effective_round_time:  # 确保不超出有效时间窗口
            break
        base_times.append(current_offset)

        # shuffled_all_request_times(base_times, start_offset, effective_round_time, time_ratio, global_start_time, experiment, buffer_time)

        for base_time in base_times:
            request_time = global_start_time + base_time
            request_times.append(request_time)

    return request_times


def shuffled_all_request_times(base_times, start_offset, effective_round_time, time_ratio, global_start_time,
                               experiment, buffer_time):
    # 对时间点进行轻微的随机打散，但保持总体顺序
    shuffled_times = []
    for i, base_offset in enumerate(base_times):
        # 添加小幅随机偏移
        jitter = random.uniform(-0.5, 0.5)  # ±0.5秒的抖动
        jittered_offset = base_offset + jitter
        jittered_offset = max(start_offset, min(effective_round_time, jittered_offset))  # 确保在有效范围内

        # 应用非线性映射（在有效时间窗口内）
        if time_ratio > 1 and jittered_offset <= effective_round_time:
            # 使用sigmoid类函数进行平滑映射
            progress = jittered_offset / effective_round_time
            # 调整后的进度，保持开始和结束点不变，但中间部分根据time_ratio拉伸
            adjusted_progress = progress ** (1 / time_ratio)
            adjusted_offset = adjusted_progress * effective_round_time
        else:
            # time_ratio <= 1的情况，直接线性缩放
            adjusted_offset = jittered_offset * time_ratio

        # 确保调整后的时间不会超出有效时间窗口
        if adjusted_offset > effective_round_time:
            adjusted_offset = effective_round_time

        # 将偏移转换为绝对时间
        request_time = global_start_time + adjusted_offset
        shuffled_times.append(request_time)

    # 最后再次打散一部分时间点，增加随机性
    if len(shuffled_times) > 1:
        # 随机选择20%的时间点进行轻微重排
        num_to_shuffle = max(1, len(shuffled_times) // 5)
        indices_to_shuffle = random.sample(range(len(shuffled_times)), num_to_shuffle)

        # 对选中的时间点进行局部随机化（确保仍在有效时间窗口内）
        for idx in indices_to_shuffle:
            # 在附近范围内随机调整时间
            if idx > 0 and idx < len(shuffled_times) - 1:
                min_time = (shuffled_times[idx - 1] + shuffled_times[idx]) / 2
                max_time = (shuffled_times[idx] + shuffled_times[idx + 1]) / 2
                # 确保不超出有效时间窗口
                max_time = min(max_time, global_start_time + effective_round_time)
                if min_time < max_time:
                    shuffled_times[idx] = random.uniform(min_time, max_time)

    # 确保时间点仍然是递增的
    shuffled_times.sort()

    # 最终检查：确保所有时间点都在有效窗口内
    shuffled_times = [t for t in shuffled_times if t <= global_start_time + effective_round_time]

    experiment.logger.info(
        f"Generated {len(shuffled_times)} request times in {effective_round_time}s window, buffer: {buffer_time}s")
    if shuffled_times:
        last_request_time = shuffled_times[-1] - global_start_time
        experiment.logger.info(
            f"Last request at: {last_request_time:.2f}s, buffer remaining: {effective_round_time - last_request_time:.2f}s")

    return shuffled_times


async def worker(experiment, selected_clients, semaphore, results, worker_id, worker_json, qmp_per_worker):
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

    # 创建线程安全的token计数器
    tokens_counter = ThreadSafeCounter()

    # 预先计算所有请求的时间点
    request_times = calculate_all_request_times(experiment, qmp_per_worker)
    client_id = f"{experiment.client_id}_worker_{worker_id}"

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
            else:
                experiment.logger.warning(
                    f"[{client_id}] Warning: Negative sleep time detected: {sleep_time:.6f} seconds")
                continue

        # 发送请求（不管是否需要sleep，都会执行到这里）
        request = random.choice(worker_json)
        selected_client = selected_clients[worker_id % len(selected_clients)]
        task = asyncio.create_task(
            process_request(selected_client, experiment, request, worker_id, results, semaphore, tokens_counter)
        )
        task_status[task] = {"start_time": time.time(), "status": "running"}
        task.add_done_callback(lambda t: task_status.update({t: {"status": "completed", "end_time": time.time()}}))
        tasks.append(task)
        request_count += 1

    elapsed = time.time() - global_start_time
    remaining_time = experiment.round_time - elapsed
    if remaining_time > experiment.round_time * GLOBAL_CONFIG.get('buffer_ratio', 0.5) * 1.1:
        experiment.logger.warning(
            f"[{client_id}] Warning: Not enough requests to fill the round time. Sleeping for {remaining_time:.2f} seconds")
        await asyncio.sleep(remaining_time)
    else:
        experiment.logger.info(f"[{client_id}] reached the end of the round time.")

    # 等待所有任务完成
    if tasks:
        # 等待取消操作完成
        # await asyncio.gather(*tasks, return_exceptions=True)

        # 计算总耗时
        total_elapsed_time = time.time() - global_start_time

        completed = sum(1 for status in task_status.values() if status["status"] == "completed")
        cancelled_count = sum(1 for task in tasks if task.cancelled())

        experiment.logger.info(f"Total tasks: {request_count}, Completed: {completed}, Cancelled: {cancelled_count}")
        experiment.logger.info(f"Task completion rate: {completed / len(tasks) * 100:.2f}%")
        experiment.logger.info(f"Total tokens processed: {tokens_counter.value}")
        experiment.logger.info(
            f"Total elapsed time: {total_elapsed_time:.2f} seconds, Round time: {experiment.round_time:.2f} seconds, More than round time: {total_elapsed_time - experiment.round_time:.2f} seconds")
        for task in tasks:
            task.cancel()

    return completed, drift_time, request_count


async def process_request(client, experiment, request, worker_id, results, semaphore, tokens_counter):
    async with semaphore:
        try:
            # 检查当前token总数是否超限
            if hasattr(experiment, 'max_tokens') and tokens_counter.value >= experiment.max_tokens:
                experiment.logger.info(f"Worker {worker_id} reached max tokens limit ({experiment.max_tokens})")
                return

            result = await make_request(client, experiment, request)
            if result:
                output_tokens = result[0]  # 第一个元素是output_tokens
                # 原子性地更新token计数
                new_total = tokens_counter.add(output_tokens)
                results.append(result)

                # 如果超过限制，记录日志
                if hasattr(experiment, 'max_tokens') and new_total >= experiment.max_tokens:
                    experiment.logger.info(
                        f"Worker {worker_id} reached max tokens after processing: {new_total}/{experiment.max_tokens}")

        except Exception as e:
            logging.error(
                f"Worker {worker_id} {experiment.config_round + 1} round for client {experiment.client_index} raised an exception: {e}")


async def worker_with_queue(experiment, queue_manager, semaphore, results, worker_id, worker_json, qmp_per_worker):
    """使用队列管理器的worker函数"""
    assert worker_json is not None, "sample_content is None!"
    assert isinstance(worker_json, list), f"sample_content is not a list! type={type(worker_json)}"
    assert len(worker_json) > 0, "sample_content is empty!"

    global_start_time = time.time()
    request_count = 0
    drift_time = 0
    completed = 0
    tasks = []
    task_status = {}

    # 创建线程安全的token计数器
    tokens_counter = ThreadSafeCounter()

    # 注册客户端到队列管理器
    client_id = f"{experiment.client_id}_worker_{worker_id}"
    # 移除直接注册，submit_request会自动处理
    # await queue_manager.register_client(client_id, experiment.client.client_type)

    # 预先计算所有请求的时间点
    request_times = calculate_all_request_times(experiment, qmp_per_worker)

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
            else:
                experiment.logger.warning(
                    f"[{client_id}] Warning: Negative sleep time detected: {sleep_time:.6f} seconds")
                continue

        # 发送请求到队列（不管是否需要sleep，都会执行到这里）
        request = random.choice(worker_json)

        # 设置优先级（短请求优先级更高）
        priority = experiment.client.priority

        task = asyncio.create_task(
            process_request_with_queue(queue_manager, client_id, experiment, request,
                                       worker_id, results, semaphore, tokens_counter,
                                       priority)
        )
        task_status[task] = {"start_time": time.time(), "status": "running"}
        task.add_done_callback(lambda t: task_status.update({t: {"status": "completed", "end_time": time.time()}}))
        tasks.append(task)
        request_count += 1

    elapsed = time.time() - global_start_time
    remaining_time = experiment.round_time - elapsed
    if remaining_time > experiment.round_time * GLOBAL_CONFIG.get('buffer_ratio', 0.5) * 1.1:  # 只在剩余时间大于3秒时才sleep，防止误差
        experiment.logger.warning(
            f"[{client_id}] Warning: Not enough requests to fill the round time. Sleeping for {remaining_time:.2f} seconds")
        await asyncio.sleep(remaining_time)
    else:
        experiment.logger.info(f"[{client_id}] reached the end of the round time.")

    # 等待所有任务完成
    if tasks:
        # await asyncio.gather(*tasks, return_exceptions=True)

        # 计算总耗时
        total_elapsed_time = time.time() - global_start_time

        completed = sum(1 for status in task_status.values() if status["status"] == "completed")
        cancelled_count = sum(1 for task in tasks if task.cancelled())

        experiment.logger.info(f"Total tasks: {request_count}, Completed: {completed}, Cancelled: {cancelled_count}")
        experiment.logger.info(f"Task completion rate: {completed / len(tasks) * 100:.2f}%")
        experiment.logger.info(f"Total tokens processed: {tokens_counter.value}")
        experiment.logger.info(
            f"Total elapsed time: {total_elapsed_time:.2f} seconds, Round time: {experiment.round_time:.2f} seconds, More than round time: {total_elapsed_time - experiment.round_time:.2f} seconds")

        for task in tasks:
            task.cancel()

    return completed, drift_time, request_count


async def process_request_with_queue(queue_manager, client_id, experiment, request, worker_id, results, semaphore,
                                     tokens_counter, priority=0):
    """使用队列管理器处理请求"""
    async with semaphore:
        try:
            # 检查当前token总数是否超限
            if hasattr(experiment, 'max_tokens') and tokens_counter.value >= experiment.max_tokens:
                experiment.logger.info(f"Worker {worker_id} reached max tokens limit ({experiment.max_tokens})")
                return

            result = await make_request_via_queue(
                queue_manager, client_id, f"worker_{worker_id}",
                request, experiment, priority
            )

            if result:
                output_tokens = result[0]  # 第一个元素是output_tokens
                # 原子性地更新token计数
                new_total = tokens_counter.add(output_tokens)
                results.append(result)

                # 如果超过限制，记录日志
                if hasattr(experiment, 'max_tokens') and new_total >= experiment.max_tokens:
                    experiment.logger.info(
                        f"Worker {worker_id} reached max tokens after processing: {new_total}/{experiment.max_tokens}")

        except Exception as e:
            logging.error(
                f"Worker {worker_id} {experiment.config_round + 1} round for client {experiment.client_index} raised an exception: {e}")

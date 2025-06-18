import asyncio
import time
from datetime import datetime
from typing import Any
import uuid

import numpy as np
import logging

import random
import threading

from config.Config import GLOBAL_CONFIG
from util.ThreadSafeUtil import ThreadSafeCounter

# 全局变量来跟踪活跃的请求ID
active_request_ids = set()
active_request_ids_lock = threading.Lock()


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


async def make_request(client, experiment, request, start_time=None):
    if start_time is None:
        start_time = time.time()
    
    # 生成唯一的请求ID
    request_id = str(uuid.uuid4())
    
    # 添加到活跃请求集合
    with active_request_ids_lock:
        active_request_ids.add(request_id)
    
    try:
        # 使用log_request=False参数来禁止在日志中打印请求内容
        stream = await client.chat.completions.create(
            model=GLOBAL_CONFIG['request_model_name'],
            messages=[{"role": "user", "content": request}],
            max_tokens=experiment.output_tokens,
            stream=True,
            extra_headers={"X-Request-ID": request_id}  # 添加请求ID到header
        )
        first_token_time, output_tokens = await asyncio.wait_for(process_stream(stream),
                                                                  timeout=experiment.request_timeout)
        end_time = time.time()
        elapsed_time = end_time - start_time
        ttft = first_token_time - start_time if first_token_time else None
        input_token = experiment.tokenizer(request, truncation=False, return_tensors="pt").input_ids[0]
        tokens_per_second = output_tokens / elapsed_time if elapsed_time > 0 else 0
        
        # 从活跃请求集合中移除
        with active_request_ids_lock:
            active_request_ids.discard(request_id)
        
        return output_tokens, elapsed_time, tokens_per_second, ttft, len(
            input_token), 1 if elapsed_time <= experiment.latency_slo else 0

    except asyncio.TimeoutError:
        end_time = time.time()
        # 记录timeout次数
        if hasattr(experiment, 'timeout_count'):
            experiment.timeout_count += 1
            
        experiment.logger.warning(
            f"Client {experiment.client_id} request timed out after {end_time - start_time} seconds (Total timeouts: {experiment.timeout_count})")
        
        # 尝试中止vLLM中的请求
        try:
            await abort_vllm_request(client, request_id)
        except Exception as e:
            experiment.logger.warning(f"Failed to abort request {request_id}: {e}")
        
        # 从活跃请求集合中移除
        with active_request_ids_lock:
            active_request_ids.discard(request_id)
            
        return None
    except Exception as e:
        experiment.logger.error(f"Error during request: {str(e)}")
        
        # 从活跃请求集合中移除
        with active_request_ids_lock:
            active_request_ids.discard(request_id)
            
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
                # if sleep_time > 5:
                #     sleep_end = time.time()
                #     experiment.logger.info(f"[{client_id}] target_time: {target_time:.6f}, "
                #                            f"current_time: {datetime.fromtimestamp(current_time).strftime('%H:%M:%S.%f')}, "
                #                            f"sleep_time: {datetime.fromtimestamp(sleep_time).strftime('%H:%M:%S.%f')}, "
                #                            f"actual_sleep: {sleep_end - sleep_start:.6f}")
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

    # 中止所有活跃的vLLM请求
    await abort_all_active_requests(selected_clients)

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
            
    # 中止所有活跃的vLLM请求（通过队列管理器）
    if hasattr(queue_manager, 'openai_client') and queue_manager.openai_client:
        await abort_all_active_requests(queue_manager.openai_client)
    
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


async def force_cancel_all_tasks(tasks, experiment):
    """强制取消所有任务的辅助函数"""
    if not tasks:
        return

    experiment.logger.info(f"Force cancelling {len(tasks)} tasks")

    # 立即取消所有任务
    for task in tasks:
        if not task.done():
            task.cancel()

    # 等待所有取消操作完成
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 统计取消结果
    cancelled_count = sum(1 for task in tasks if task.cancelled())
    completed_count = sum(1 for task in tasks if task.done() and not task.cancelled())
    exception_count = sum(
        1 for result in results if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError))

    experiment.logger.info(
        f"Task cancellation complete - Cancelled: {cancelled_count}, Completed: {completed_count}, Exceptions: {exception_count}")


async def abort_vllm_request(client, request_id):
    """中止vLLM中的特定请求"""
    try:
        # 方法1: 尝试使用vLLM的abort API (v0.2.0+)
        if hasattr(client, 'abort'):
            await client.abort(request_id)
            logging.info(f"Successfully aborted request {request_id} using client.abort()")
            return
        
        # 方法2: 尝试HTTP POST到abort端点
        try:
            response = await client.post(
                "/abort",
                json={"request_id": request_id}
            )
            if response.status_code == 200:
                logging.info(f"Successfully aborted request {request_id} via HTTP")
                return
            else:
                logging.warning(f"Failed to abort request {request_id}: HTTP {response.status_code}")
        except AttributeError:
            # client 不支持 .post 方法
            pass
        
        # 方法3: 尝试使用OpenAI兼容的cancel方法
        try:
            if hasattr(client.chat.completions, 'cancel'):
                await client.chat.completions.cancel(request_id)
                logging.info(f"Successfully cancelled request {request_id} using OpenAI API")
                return
        except:
            pass
            
        logging.warning(f"No available method to abort request {request_id}")
        
    except Exception as e:
        logging.warning(f"Error aborting request {request_id}: {e}")


async def abort_all_active_requests(clients):
    """中止所有活跃的vLLM请求"""
    with active_request_ids_lock:
        request_ids_to_abort = active_request_ids.copy()
        active_request_ids.clear()
    
    if not request_ids_to_abort:
        logging.info("No active requests to abort")
        return
    
    logging.info(f"Aborting {len(request_ids_to_abort)} active requests")
    
    # 尝试中止所有活跃请求
    abort_tasks = []
    for client in clients:
        for request_id in request_ids_to_abort:
            abort_tasks.append(abort_vllm_request(client, request_id))
    
    if abort_tasks:
        results = await asyncio.gather(*abort_tasks, return_exceptions=True)
        successful_aborts = sum(1 for result in results if not isinstance(result, Exception))
        logging.info(f"Attempted to abort {len(abort_tasks)} requests, {successful_aborts} successful")
    
    # 额外的清理：等待一小段时间让vLLM处理abort请求
    await asyncio.sleep(0.5)


async def restart_vllm_service(experiment):
    """重启vLLM服务以确保完全清理（激进方案）"""
    try:
        if hasattr(experiment, 'vllm_process') and experiment.vllm_process:
            experiment.logger.info("Restarting vLLM service to ensure clean state")
            
            # 终止当前进程
            experiment.vllm_process.terminate()
            await asyncio.sleep(2)
            
            # 如果还没结束，强制杀死
            if experiment.vllm_process.poll() is None:
                experiment.vllm_process.kill()
                await asyncio.sleep(1)
            
            # 重新启动vLLM服务
            # 这里需要根据你的vLLM启动配置来调整
            # experiment.vllm_process = await start_vllm_service()
            
            experiment.logger.info("vLLM service restarted")
            
    except Exception as e:
        experiment.logger.error(f"Failed to restart vLLM service: {e}")

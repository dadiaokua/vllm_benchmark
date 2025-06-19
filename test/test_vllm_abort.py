#!/usr/bin/env python3
"""
测试vLLM AsyncLLMEngine的abort功能

这个脚本用于测试：
1. 能否启动AsyncLLMEngine
2. 能否发送请求
3. 能否中断/abort正在进行的请求
4. abort方法的可用性和效果
"""

import asyncio
import time
import uuid
import logging
from typing import List, Optional

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_vllm_abort():
    """测试vLLM的abort功能"""
    try:
        from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
        
        logger.info("=== 开始测试vLLM AsyncLLMEngine abort功能 ===")
        
        # 1. 创建引擎参数（使用较小的模型进行测试）
        logger.info("1. 创建引擎参数...")
        engine_args = AsyncEngineArgs(
            model="/home/llm/model_hub/Llama-3.1-8B",
            max_num_seqs=4,
            gpu_memory_utilization=0.3,  # 使用较少内存
            tensor_parallel_size=1,
            trust_remote_code=True,
            enable_chunked_prefill=False,
            dtype='float16',
        )
        
        # 2. 创建引擎
        logger.info("2. 创建AsyncLLMEngine...")
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        logger.info("✓ AsyncLLMEngine创建成功")
        
        # 3. 准备测试数据
        test_prompt = "Please write a very long story about artificial intelligence and machine learning. Include detailed explanations of neural networks, deep learning algorithms, and their applications in modern technology. Make it at least 1000 words long and very detailed."
        
        sampling_params = SamplingParams(
            max_tokens=500,  # 生成较多token以便有时间abort
            temperature=0.7,
            top_p=0.9
        )
        
        # 4. 测试正常请求
        logger.info("3. 测试正常请求...")
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        results = []
        async for output in engine.generate(test_prompt, sampling_params, request_id):
            results.append(output)
            if len(results) >= 3:  # 收集几个输出后就停止
                break
                
        logger.info(f"✓ 正常请求完成，耗时: {time.time() - start_time:.2f}s，输出数量: {len(results)}")
        
        # 5. 测试abort功能
        logger.info("4. 测试abort功能...")
        await test_abort_methods(engine)
        
        # 6. 测试多并发请求和abort
        logger.info("5. 测试多并发请求和abort...")
        await test_concurrent_abort(engine, sampling_params)
        
        logger.info("=== 所有测试完成 ===")
        
    except ImportError as e:
        logger.error(f"❌ 无法导入vLLM: {e}")
        logger.error("请确保已安装vLLM: pip install vllm")
    except Exception as e:
        logger.error(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


async def test_abort_methods(engine):
    """测试各种可能的abort方法"""
    logger.info("4.1 检查engine对象的可用方法...")
    
    # 检查engine对象有哪些方法
    engine_methods = [method for method in dir(engine) if not method.startswith('_')]
    logger.info(f"Engine可用方法: {engine_methods}")
    
    # 检查是否有abort相关方法
    abort_methods = [method for method in engine_methods if 'abort' in method.lower()]
    logger.info(f"Abort相关方法: {abort_methods}")
    
    # 检查是否有cancel相关方法
    cancel_methods = [method for method in engine_methods if 'cancel' in method.lower()]
    logger.info(f"Cancel相关方法: {cancel_methods}")
    
    # 检查是否有stop相关方法
    stop_methods = [method for method in engine_methods if 'stop' in method.lower()]
    logger.info(f"Stop相关方法: {stop_methods}")
    
    # 检查engine.engine（内部引擎）的方法
    if hasattr(engine, 'engine'):
        logger.info("4.2 检查内部engine的方法...")
        inner_engine_methods = [method for method in dir(engine.engine) if not method.startswith('_')]
        logger.info(f"内部Engine可用方法: {inner_engine_methods}")
        
        inner_abort_methods = [method for method in inner_engine_methods if 'abort' in method.lower()]
        logger.info(f"内部Engine Abort相关方法: {inner_abort_methods}")


async def test_concurrent_abort(engine, sampling_params):
    """测试并发请求和abort"""
    logger.info("5.1 启动多个并发请求...")
    
    # 创建多个长请求
    prompts = [
        "Write a detailed technical explanation of quantum computing with at least 800 words.",
        "Explain the history and development of artificial intelligence in great detail.",
        "Describe the principles of blockchain technology with comprehensive examples.",
    ]
    
    request_ids = []
    tasks = []
    
    # 启动多个异步生成任务
    for i, prompt in enumerate(prompts):
        request_id = f"test_request_{i}_{uuid.uuid4()}"
        request_ids.append(request_id)
        
        # 创建异步任务
        task = asyncio.create_task(collect_generation_output(engine, prompt, sampling_params, request_id))
        tasks.append(task)
        
        logger.info(f"启动请求 {i+1}: {request_id}")
    
    # 等待一小段时间让请求开始处理
    await asyncio.sleep(2)
    
    logger.info("5.2 尝试取消部分请求...")
    
    # 尝试取消第一个任务
    if tasks:
        logger.info(f"取消第一个任务...")
        tasks[0].cancel()
        
        try:
            await tasks[0]
        except asyncio.CancelledError:
            logger.info("✓ 第一个任务已成功取消")
        except Exception as e:
            logger.warning(f"取消第一个任务时出现异常: {e}")
    
    # 等待其余任务完成或超时
    logger.info("5.3 等待其余任务完成...")
    remaining_tasks = tasks[1:]
    
    if remaining_tasks:
        try:
            # 设置超时时间
            await asyncio.wait_for(asyncio.gather(*remaining_tasks, return_exceptions=True), timeout=30)
            logger.info("✓ 其余任务已完成")
        except asyncio.TimeoutError:
            logger.warning("⚠️ 其余任务超时，强制取消...")
            for task in remaining_tasks:
                if not task.done():
                    task.cancel()


async def collect_generation_output(engine, prompt: str, sampling_params, request_id: str) -> List:
    """收集生成输出"""
    results = []
    try:
        logger.info(f"开始处理请求: {request_id}")
        async for output in engine.generate(prompt, sampling_params, request_id):
            results.append(output)
            # 每收集到一个输出就记录一下
            if len(results) % 5 == 0:
                logger.info(f"请求 {request_id} 已生成 {len(results)} 个输出")
                
        logger.info(f"✓ 请求 {request_id} 完成，总输出: {len(results)}")
        return results
        
    except asyncio.CancelledError:
        logger.info(f"✓ 请求 {request_id} 被取消，已生成: {len(results)} 个输出")
        raise
    except Exception as e:
        logger.error(f"❌ 请求 {request_id} 出现错误: {e}")
        return results


async def test_engine_abort_apis(engine, request_id: str):
    """测试引擎的各种abort API"""
    logger.info(f"6. 测试引擎abort API for request {request_id}...")
    
    # 方法1: 检查是否有abort_request方法
    if hasattr(engine, 'abort_request'):
        try:
            logger.info("尝试调用 engine.abort_request()...")
            result = await engine.abort_request(request_id)
            logger.info(f"✓ abort_request 成功: {result}")
            return True
        except Exception as e:
            logger.warning(f"abort_request 失败: {e}")
    
    # 方法2: 检查内部引擎的abort方法
    if hasattr(engine, 'engine') and hasattr(engine.engine, 'abort_request'):
        try:
            logger.info("尝试调用 engine.engine.abort_request()...")
            result = engine.engine.abort_request(request_id)
            logger.info(f"✓ engine.abort_request 成功: {result}")
            return True
        except Exception as e:
            logger.warning(f"engine.abort_request 失败: {e}")
    
    # 方法3: 检查其他可能的abort方法
    abort_method_names = ['abort', 'cancel_request', 'stop_request', 'terminate_request']
    
    for method_name in abort_method_names:
        if hasattr(engine, method_name):
            try:
                logger.info(f"尝试调用 engine.{method_name}()...")
                method = getattr(engine, method_name)
                if asyncio.iscoroutinefunction(method):
                    result = await method(request_id)
                else:
                    result = method(request_id)
                logger.info(f"✓ {method_name} 成功: {result}")
                return True
            except Exception as e:
                logger.warning(f"{method_name} 失败: {e}")
    
    logger.warning("❌ 未找到可用的abort方法")
    return False


if __name__ == "__main__":
    # 运行测试
    asyncio.run(test_vllm_abort()) 
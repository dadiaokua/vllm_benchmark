#!/usr/bin/env python3
"""
测试vLLM引擎的请求优先级和队列调度功能

测试内容：
1. 请求优先级设置
2. 队列调度策略配置
3. 高优先级请求插队效果
4. 不同scheduling_policy的测试
"""

import asyncio
import uuid
import time
import logging
from typing import List
from vllm_engine_helper import VLLMEngineManager, create_sampling_params

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_priority_scheduling():
    """测试优先级调度功能"""
    logger.info("=== 测试vLLM优先级调度功能 ===")
    
    engine_manager = VLLMEngineManager()
    
    try:
        # 启动引擎，启用优先级调度
        logger.info("启动引擎并配置优先级调度...")
        engine = await engine_manager.start_engine(
            scheduling_policy="priority",  # 启用优先级调度
            max_num_seqs=4
        )
        
        sampling_params = create_sampling_params(max_tokens=100)
        
        # 测试优先级调度
        await test_priority_queue_order(engine, sampling_params)
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await engine_manager.shutdown_engine()

async def test_priority_queue_order(engine, sampling_params):
    """测试不同优先级的请求处理顺序"""
    logger.info("1. 测试请求优先级队列顺序...")
    
    # 创建不同优先级的请求
    requests = []
    
    # 先添加低优先级请求（priority=0, 默认值）
    for i in range(3):
        request_id = f"low_priority_{i}_{uuid.uuid4()}"
        prompt = f"Low priority request {i}: Write a short story about cats."
        
        logger.info(f"添加低优先级请求: {request_id}")
        # 使用engine.generate方法，并且查看是否支持priority参数
        
        task = asyncio.create_task(
            collect_generation_with_priority(engine, prompt, sampling_params, request_id, priority=0)
        )
        requests.append(("low", request_id, task))
        
        # 稍微延迟以确保请求按顺序提交
        await asyncio.sleep(0.1)
    
    # 稍等片刻让低优先级请求开始排队
    await asyncio.sleep(0.5)
    
    # 添加高优先级请求
    for i in range(2):
        request_id = f"high_priority_{i}_{uuid.uuid4()}"
        prompt = f"High priority request {i}: Quickly explain quantum physics."
        
        logger.info(f"添加高优先级请求: {request_id}")
        task = asyncio.create_task(
            collect_generation_with_priority(engine, prompt, sampling_params, request_id, priority=10)
        )
        requests.append(("high", request_id, task))
        
        await asyncio.sleep(0.1)
    
    # 等待所有请求完成并记录完成顺序
    completion_order = []
    completed_tasks = []
    
    while len(completed_tasks) < len(requests):
        for priority_type, request_id, task in requests:
            if task not in completed_tasks and task.done():
                try:
                    result = await task
                    completion_order.append(f"{priority_type}_{request_id}")
                    completed_tasks.append(task)
                    logger.info(f"✓ 请求完成: {priority_type} - {request_id}")
                except Exception as e:
                    logger.warning(f"请求异常: {request_id} - {e}")
                    completed_tasks.append(task)
        
        await asyncio.sleep(0.1)
    
    logger.info(f"请求完成顺序: {completion_order}")
    
    # 分析结果
    logger.info("2. 分析优先级调度效果...")
    high_priority_positions = []
    low_priority_positions = []
    
    for i, request in enumerate(completion_order):
        if request.startswith("high"):
            high_priority_positions.append(i)
        else:
            low_priority_positions.append(i)
    
    logger.info(f"高优先级请求完成位置: {high_priority_positions}")
    logger.info(f"低优先级请求完成位置: {low_priority_positions}")
    
    # 检查是否高优先级请求优先完成
    if high_priority_positions and low_priority_positions:
        avg_high_pos = sum(high_priority_positions) / len(high_priority_positions)
        avg_low_pos = sum(low_priority_positions) / len(low_priority_positions)
        
        if avg_high_pos < avg_low_pos:
            logger.info("✓ 优先级调度有效！高优先级请求平均更早完成")
        else:
            logger.warning("⚠️ 优先级调度效果不明显")

async def collect_generation_with_priority(engine, prompt: str, sampling_params, request_id: str, priority: int = 0):
    """收集生成输出，支持优先级"""
    results = []
    start_time = time.time()
    
    try:
        logger.info(f"开始处理请求 (优先级={priority}): {request_id}")
        
        # 检查engine.generate方法是否支持priority参数
        import inspect
        gen_signature = inspect.signature(engine.generate)
        supports_priority = 'priority' in gen_signature.parameters
        
        if supports_priority:
            logger.info(f"引擎支持priority参数，设置优先级: {priority}")
            async for output in engine.generate(prompt, sampling_params, request_id, priority=priority):
                results.append(output)
        else:
            logger.warning("引擎不支持priority参数，使用默认优先级")
            async for output in engine.generate(prompt, sampling_params, request_id):
                results.append(output)
                
        end_time = time.time()
        logger.info(f"✓ 请求完成: {request_id} (优先级={priority}), 耗时: {end_time-start_time:.2f}s, 输出: {len(results)}")
        return results
        
    except asyncio.CancelledError:
        logger.info(f"✓ 请求被取消: {request_id} (优先级={priority}), 已生成: {len(results)}")
        return results
    except Exception as e:
        logger.error(f"❌ 请求出错: {request_id} (优先级={priority}) - {e}")
        return results

async def test_scheduling_policies():
    """测试不同的调度策略"""
    logger.info("=== 测试不同调度策略 ===")
    
    policies = ["fifo", "priority"]  # 可能支持的调度策略
    
    for policy in policies:
        logger.info(f"测试调度策略: {policy}")
        engine_manager = VLLMEngineManager()
        
        try:
            # 尝试设置不同的调度策略
            engine = await engine_manager.start_engine(
                scheduling_policy=policy,
                max_num_seqs=2
            )
            
            logger.info(f"✓ 成功启动引擎，调度策略: {policy}")
            
            # 简单测试
            sampling_params = create_sampling_params(max_tokens=20)
            request_id = f"test_{policy}_{uuid.uuid4()}"
            
            results = []
            async for output in engine.generate("Test prompt", sampling_params, request_id):
                results.append(output)
                break  # 只获取第一个输出
            
            logger.info(f"✓ 策略 {policy} 测试成功")
            
        except Exception as e:
            logger.warning(f"策略 {policy} 不支持或测试失败: {e}")
        finally:
            await engine_manager.shutdown_engine()

async def test_engine_methods_inspection():
    """检查引擎支持的方法和参数"""
    logger.info("=== 检查引擎方法和参数 ===")
    
    engine_manager = VLLMEngineManager()
    
    try:
        engine = await engine_manager.start_engine()
        
        # 检查generate方法的签名
        import inspect
        
        logger.info("1. 检查engine.generate方法签名...")
        gen_signature = inspect.signature(engine.generate)
        logger.info(f"generate方法参数: {list(gen_signature.parameters.keys())}")
        
        # 检查add_request方法的签名（如果存在）
        if hasattr(engine, 'add_request'):
            logger.info("2. 检查engine.add_request方法签名...")
            add_req_signature = inspect.signature(engine.add_request)
            logger.info(f"add_request方法参数: {list(add_req_signature.parameters.keys())}")
        
        # 检查引擎配置
        if hasattr(engine, 'engine'):
            if hasattr(engine.engine, 'scheduler_config'):
                scheduler_config = engine.engine.scheduler_config
                logger.info("3. 检查调度器配置...")
                scheduler_attrs = [attr for attr in dir(scheduler_config) if not attr.startswith('_')]
                logger.info(f"调度器配置属性: {scheduler_attrs}")
                
                # 检查调度策略相关属性
                if hasattr(scheduler_config, 'policy'):
                    logger.info(f"当前调度策略: {scheduler_config.policy}")
                if hasattr(scheduler_config, 'scheduling_policy'):
                    logger.info(f"调度策略: {scheduler_config.scheduling_policy}")
    
    except Exception as e:
        logger.error(f"检查失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await engine_manager.shutdown_engine()

if __name__ == "__main__":
    async def main():
        # 检查引擎方法和参数
        await test_engine_methods_inspection()
        
        # 测试不同调度策略
        await test_scheduling_policies()
        
        # 测试优先级调度
        await test_priority_scheduling()
    
    asyncio.run(main()) 
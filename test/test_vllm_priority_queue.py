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
from typing import List, Union, Any
from vllm_engine_helper import VLLMEngineManager, create_sampling_params
from prompt_loader import PromptLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def safe_len(obj: Any) -> int:
    """安全地计算对象长度，避免类型错误"""
    if isinstance(obj, BaseException):
        return -1
    try:
        return len(obj) if hasattr(obj, '__len__') else 0
    except (TypeError, AttributeError):
        return 0

async def test_priority_scheduling():
    """测试优先级调度功能"""
    logger.info("=== 测试vLLM优先级调度功能 ===")

    engine_manager = VLLMEngineManager()
    prompt_loader = PromptLoader()

    try:
        # 启动引擎，启用优先级调度
        logger.info("启动引擎并配置优先级调度...")
        engine = await engine_manager.start_engine(
            scheduling_policy="priority",  # 启用优先级调度
            max_num_seqs=4,  # 从4减少到2，降低并发数
            tensor_parallel_size=8  # 明确指定单GPU
        )

        sampling_params = create_sampling_params(max_tokens=50)

        # 测试优先级调度
        await test_priority_queuing(engine, sampling_params, prompt_loader)

        # 测试多层次优先级
        logger.info("=== 测试多层次优先级 ===")
        await test_multi_level_priority(engine, sampling_params, prompt_loader)

    except Exception as e:
        logger.error(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await engine_manager.shutdown_engine()


async def test_priority_queuing(engine, sampling_params, prompt_loader):
    """测试请求优先级队列功能"""
    logger.info("=== 测试请求优先级队列 ===")

    # 获取随机prompts
    prompts = prompt_loader.get_random_prompts(15, "short")  # 从50减少到15
    if not prompts:
        # 如果没有prompts，使用默认prompts
        prompts = [
            "Write a short story about a cat",
            "Explain quantum physics simply",
            "Create a recipe for chocolate cake",
            "Describe the process of photosynthesis",
            "Write a poem about the ocean"
        ]

    # 1. 添加低优先级请求
    logger.info("1. 添加低优先级请求...")
    low_priority_tasks = []

    for i, prompt in enumerate(prompts[:8]):  # 从30减少到8个低优先级请求
        request_id = f"low_priority_{i + 1}_{uuid.uuid4()}"

        # 检查engine.generate是否支持priority参数
        try:
            task = asyncio.create_task(
                generate_with_priority(engine, prompt, sampling_params, request_id, priority=10)  # 低优先级使用大数字
            )
            low_priority_tasks.append((task, request_id))
            logger.info(f"添加低优先级请求 (priority=10): {request_id}")
            await asyncio.sleep(0.5)  # 间隔添加
        except Exception as e:
            logger.warning(f"无法设置优先级，使用普通generate: {e}")
            task = asyncio.create_task(
                generate_without_priority(engine, prompt, sampling_params, request_id)
            )
            low_priority_tasks.append((task, request_id))

    # 2. 等待一段时间，然后添加高优先级请求
    await asyncio.sleep(1)
    logger.info("2. 添加高优先级请求（应该插队）...")

    high_priority_prompt = prompts[-1] if len(prompts) > 3 else "This is a high priority urgent request"
    high_priority_id = f"high_priority_{uuid.uuid4()}"

    try:
        high_priority_task = asyncio.create_task(
            generate_with_priority(engine, high_priority_prompt, sampling_params, high_priority_id, priority=1)
            # 高优先级使用小数字
        )
        logger.info(f"添加高优先级请求 (priority=1): {high_priority_id}")
    except Exception as e:
        logger.warning(f"无法设置高优先级，使用普通generate: {e}")
        high_priority_task = asyncio.create_task(
            generate_without_priority(engine, high_priority_prompt, sampling_params, high_priority_id)
        )

    # 3. 等待所有任务完成
    logger.info("3. 等待所有请求完成...")

    try:
        # 等待所有任务完成
        all_tasks = [task for task, _ in low_priority_tasks] + [high_priority_task]
        await asyncio.gather(*all_tasks, return_exceptions=True)

        logger.info("✓ 优先级队列测试完成")

    except Exception as e:
        logger.error(f"优先级队列测试失败: {e}")


async def test_multi_level_priority(engine, sampling_params, prompt_loader):
    """测试多层次优先级功能"""
    logger.info("=== 测试多层次优先级 ===")

    # 获取多个随机prompts
    prompts = prompt_loader.get_random_prompts(25, "short")  # 从60减少到25
    if not prompts:
        prompts = [
            f"Priority test prompt {i}: Explain topic {i}" for i in range(6)
        ]

    # 定义不同优先级层次 (数字越小优先级越高)
    priority_configs = [
        (1, "最高优先级"),  # 1 = 最高优先级
        (3, "高优先级"),  # 3 = 高优先级
        (5, "中等优先级"),  # 5 = 中等优先级
        (7, "低优先级"),  # 7 = 低优先级
        (10, "最低优先级"),  # 10 = 最低优先级
    ]

    tasks = []
    request_info = []

    # 按照相反顺序添加请求（先添加低优先级，后添加高优先级）
    logger.info("1. 按相反顺序添加不同优先级的请求...")

    # 先添加低优先级请求
    for i, (priority, desc) in enumerate(reversed(priority_configs)):
        if i >= len(priority_configs):
            break
        
        # 每个优先级层次添加3个请求（从10减少到3）
        for j in range(3):
            prompt_index = i * 3 + j  # 相应调整索引计算
            if prompt_index >= len(prompts):
                break
                
            prompt = prompts[prompt_index]
            request_id = f"priority_{priority}_{j}_{uuid.uuid4()}"
            
            try:
                task = asyncio.create_task(
                    generate_with_priority_and_timing(engine, prompt, sampling_params, request_id, priority)
                )
                tasks.append(task)
                request_info.append((request_id, priority, desc))
                
                logger.info(f"添加请求 {desc} (priority={priority}): {request_id}")
                await asyncio.sleep(0.1)  # 缩短间隔
                
            except Exception as e:
                logger.warning(f"无法设置优先级 {priority}: {e}")

    # 2. 等待一段时间，然后添加紧急请求
    await asyncio.sleep(1)

    if len(prompts) > len(priority_configs):
        urgent_prompt = prompts[-1]
        urgent_id = f"urgent_0_{uuid.uuid4()}"

        try:
            urgent_task = asyncio.create_task(
                generate_with_priority_and_timing(engine, urgent_prompt, sampling_params, urgent_id, 0)  # 0 = 超高优先级
            )
            tasks.append(urgent_task)
            request_info.append((urgent_id, 0, "紧急请求"))

            logger.info(f"添加紧急请求 (priority=0): {urgent_id}")

        except Exception as e:
            logger.warning(f"无法设置紧急优先级: {e}")

    # 3. 等待所有任务完成并分析执行顺序
    logger.info("2. 等待所有请求完成并分析执行顺序...")

    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 分析结果
        logger.info("3. 分析优先级执行效果...")
        
        completed_order = []
        for i, (result, (request_id, priority, desc)) in enumerate(zip(results, request_info)):
            if isinstance(result, Exception):
                logger.warning(f"请求 {desc} 异常: {result}")
                completed_order.append((desc, priority, -1))  # 用-1表示异常
            else:
                # 使用安全的长度计算函数
                output_count = safe_len(result)
                completed_order.append((desc, priority, output_count))

        logger.info("完成顺序分析:")
        for i, (desc, priority, output_count) in enumerate(completed_order, 1):
            if output_count == -1:
                logger.info(f"  {i}. {desc} (priority={priority}) - 异常")
            else:
                logger.info(f"  {i}. {desc} (priority={priority}) - 输出: {output_count}")
        
        # 检查优先级是否按预期工作
        valid_priorities = [priority for _, priority, output_count in completed_order if output_count != -1]
        if valid_priorities and valid_priorities == sorted(valid_priorities):  # 数字越小优先级越高，应该按升序完成
            logger.info("✓ 优先级调度工作正常！请求按优先级顺序完成")
        else:
            logger.warning("⚠️ 优先级调度效果不明显，可能引擎不支持或配置问题")
            logger.info(f"实际优先级顺序: {valid_priorities}")
            logger.info(f"期望优先级顺序: {sorted(valid_priorities)}")

    except Exception as e:
        logger.error(f"多层次优先级测试失败: {e}")


async def generate_with_priority(engine, prompt, sampling_params, request_id, priority=0) -> List[Any]:
    """尝试使用优先级生成（如果支持的话）"""
    results: List[Any] = []
    start_time = time.time()

    logger.info(f"开始处理请求: {request_id} (优先级: {priority})")

    try:
        # 尝试使用priority参数
        async for output in engine.generate(prompt, sampling_params, request_id, priority=priority):
            results.append(output)
            if len(results) % 10 == 0:  # 从20改回10
                logger.info(f"请求 {request_id} 已生成 {len(results)} 个输出")

        elapsed = time.time() - start_time
        logger.info(f"✓ 请求 {request_id} 完成，耗时: {elapsed:.2f}s，输出: {len(results)}")
        return results

    except TypeError as e:
        # 如果不支持priority参数，回退到普通generate
        logger.warning(f"engine.generate不支持priority参数: {e}")
        return await generate_without_priority(engine, prompt, sampling_params, request_id)


async def generate_without_priority(engine, prompt, sampling_params, request_id) -> List[Any]:
    """不使用优先级的普通生成"""
    results: List[Any] = []
    start_time = time.time()

    logger.info(f"开始处理请求: {request_id} (普通模式)")

    try:
        async for output in engine.generate(prompt, sampling_params, request_id):
            results.append(output)
            if len(results) % 10 == 0:  # 从20改回10
                logger.info(f"请求 {request_id} 已生成 {len(results)} 个输出")

        elapsed = time.time() - start_time
        logger.info(f"✓ 请求 {request_id} 完成，耗时: {elapsed:.2f}s，输出: {len(results)}")
        return results

    except Exception as e:
        logger.error(f"❌ 请求 {request_id} 失败: {e}")
        return results


async def generate_with_priority_and_timing(engine, prompt, sampling_params, request_id, priority) -> List[Any]:
    """带时间记录的优先级生成"""
    import time

    results: List[Any] = []
    start_time = time.time()

    logger.info(f"[{time.strftime('%H:%M:%S')}] 开始处理请求: {request_id} (priority={priority})")

    try:
        async for output in engine.generate(prompt, sampling_params, request_id, priority=priority):
            results.append(output)
            if len(results) % 5 == 0:  # 从3改为5，减少日志
                logger.info(f"[{time.strftime('%H:%M:%S')}] 请求 {request_id} 已生成 {len(results)} 个输出")

        elapsed = time.time() - start_time
        logger.info(
            f"[{time.strftime('%H:%M:%S')}] ✓ 请求 {request_id} (priority={priority}) 完成，耗时: {elapsed:.2f}s，输出: {len(results)}")
        return results

    except TypeError as e:
        logger.warning(f"engine.generate不支持priority参数: {e}")
        return await generate_without_priority(engine, prompt, sampling_params, request_id)
    except Exception as e:
        logger.error(f"❌ 请求 {request_id} 失败: {e}")
        return results


async def main():
    """主函数"""
    await test_priority_scheduling()


if __name__ == "__main__":
    asyncio.run(main())

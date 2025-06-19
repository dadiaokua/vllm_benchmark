#!/usr/bin/env python3
"""
简化的vLLM abort功能测试
"""

import asyncio
import uuid
import logging
from vllm_engine_helper import VLLMEngineManager, create_sampling_params
from prompt_loader import PromptLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_abort_functionality():
    """测试abort功能"""
    engine_manager = VLLMEngineManager()
    prompt_loader = PromptLoader()
    
    try:
        # 启动引擎
        engine = await engine_manager.start_engine()
        sampling_params = create_sampling_params(max_tokens=200)
        
        # 测试abort功能
        logger.info("=== 测试abort功能 ===")
        
        # 获取随机prompt
        test_prompt = prompt_loader.get_random_prompt("long")
        if not test_prompt:
            test_prompt = "Write a detailed explanation of machine learning with mathematical formulations and examples that is very long and comprehensive."
        
        request_id = f"abort_test_{uuid.uuid4()}"
        
        logger.info(f"启动请求: {request_id}")
        logger.info(f"使用prompt: {test_prompt[:100]}...")
        
        # 异步生成任务
        generation_task = asyncio.create_task(
            collect_generation_output(engine, test_prompt, sampling_params, request_id)
        )
        
        # 等待一段时间，然后尝试取消
        await asyncio.sleep(3)
        
        logger.info("尝试abort请求...")
        success = engine_manager.abort_request(request_id)
        
        if success:
            logger.info("✓ abort请求调用成功")
        else:
            logger.warning("❌ abort请求调用失败")
        
        # 等待任务完成或被取消
        try:
            result = await asyncio.wait_for(generation_task, timeout=10)
            logger.info(f"生成任务完成，生成了 {len(result)} 个输出")
        except asyncio.TimeoutError:
            logger.info("生成任务超时")
            generation_task.cancel()
        except asyncio.CancelledError:
            logger.info("✓ 生成任务被成功取消")
        except Exception as e:
            logger.info(f"生成任务异常: {e}")
        
        logger.info("=== abort功能测试完成 ===")
        
    except Exception as e:
        logger.error(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await engine_manager.shutdown_engine()

async def collect_generation_output(engine, prompt: str, sampling_params, request_id: str):
    """收集生成输出"""
    results = []
    try:
        logger.info(f"开始处理请求: {request_id}")
        async for output in engine.generate(prompt, sampling_params, request_id):
            results.append(output)
            # 每收集到一些输出就记录一下
            if len(results) % 10 == 0:
                logger.info(f"请求 {request_id} 已生成 {len(results)} 个输出")
                
        logger.info(f"✓ 请求 {request_id} 完成，总输出: {len(results)}")
        return results
        
    except asyncio.CancelledError:
        logger.info(f"✓ 请求 {request_id} 被取消，已生成: {len(results)} 个输出")
        return results  # 返回已生成的结果，不再抛出异常
    except Exception as e:
        logger.error(f"❌ 请求 {request_id} 出现错误: {e}")
        return results

async def test_multiple_abort():
    """测试多个请求的abort功能"""
    logger.info("=== 测试多个请求abort ===")
    
    engine_manager = VLLMEngineManager()
    prompt_loader = PromptLoader()
    
    try:
        engine = await engine_manager.start_engine(max_num_seqs=3)
        sampling_params = create_sampling_params(max_tokens=150)
        
        # 获取多个随机prompts
        prompts = prompt_loader.get_random_prompts(4, "mixed")
        if not prompts:
            prompts = [
                "Explain artificial intelligence in detail",
                "Write a comprehensive guide to quantum computing", 
                "Describe the history of space exploration",
                "Create a detailed recipe for French cuisine"
            ]
        
        # 启动多个请求
        tasks = []
        request_ids = []
        
        for i, prompt in enumerate(prompts):
            request_id = f"multi_test_{i}_{uuid.uuid4()}"
            request_ids.append(request_id)
            
            task = asyncio.create_task(
                collect_generation_output(engine, prompt, sampling_params, request_id)
            )
            tasks.append(task)
            
            logger.info(f"启动请求 {i+1}: {request_id}")
            await asyncio.sleep(0.5)  # 间隔启动
        
        # 等待一段时间，然后abort部分请求
        await asyncio.sleep(2)
        
        # abort前两个请求
        for i in range(2):
            logger.info(f"尝试abort请求: {request_ids[i]}")
            success = engine_manager.abort_request(request_ids[i])
            logger.info(f"abort结果: {'成功' if success else '失败'}")
        
        # 等待所有任务完成
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info("=== 多个请求abort测试完成 ===")
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.info(f"请求 {i+1} 异常: {result}")
            else:
                logger.info(f"请求 {i+1} 完成: {len(result)} 个输出")
                
    except Exception as e:
        logger.error(f"多请求abort测试失败: {e}")
    finally:
        await engine_manager.shutdown_engine()

async def main():
    """主函数"""
    # 测试单个请求abort
    await test_abort_functionality()
    
    # 测试多个请求abort
    await test_multiple_abort()

if __name__ == "__main__":
    asyncio.run(main()) 
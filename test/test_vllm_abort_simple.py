#!/usr/bin/env python3
"""
简化的vLLM abort功能测试
"""

import asyncio
import uuid
import logging
from vllm_engine_helper import VLLMEngineManager, create_sampling_params

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_abort_functionality():
    """测试abort功能"""
    engine_manager = VLLMEngineManager()
    
    try:
        # 启动引擎
        engine = await engine_manager.start_engine()
        sampling_params = create_sampling_params(max_tokens=200)
        
        # 测试abort功能
        logger.info("=== 测试abort功能 ===")
        
        # 启动一个长请求
        test_prompt = "Write a detailed explanation of machine learning with mathematical formulations and examples."
        request_id = f"abort_test_{uuid.uuid4()}"
        
        logger.info(f"启动请求: {request_id}")
        
        # 异步生成任务
        async def generate_with_abort():
            results = []
            try:
                async for output in engine.generate(test_prompt, sampling_params, request_id):
                    results.append(output)
                    if len(results) % 10 == 0:
                        logger.info(f"已生成 {len(results)} 个输出")
                return results
            except asyncio.CancelledError:
                logger.info(f"✓ 请求被取消，已生成: {len(results)} 个输出")
                return results
        
        generation_task = asyncio.create_task(generate_with_abort())
        
        # 等待一段时间后abort
        await asyncio.sleep(1)
        
        # 测试abort
        logger.info(f"调用abort: {request_id}")
        success = engine_manager.abort_request(request_id)
        
        if success:
            logger.info("✓ abort调用成功")
        else:
            logger.error("❌ abort调用失败")
        
        # 等待任务完成
        results = await generation_task
        logger.info(f"最终生成了 {len(results)} 个输出")
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await engine_manager.shutdown_engine()

if __name__ == "__main__":
    asyncio.run(test_abort_functionality()) 
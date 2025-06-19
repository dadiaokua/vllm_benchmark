#!/usr/bin/env python3
"""
简化的vLLM AsyncLLMEngine测试

主要检查：
1. vLLM是否正确安装
2. AsyncLLMEngine API的可用性
3. abort相关方法的存在性
"""

import asyncio
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_vllm_api_availability():
    """测试vLLM API的可用性"""
    try:
        logger.info("=== 测试vLLM API可用性 ===")
        
        # 1. 测试导入
        logger.info("1. 测试导入vLLM模块...")
        try:
            from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
            logger.info("✓ vLLM模块导入成功")
        except ImportError as e:
            logger.error(f"❌ vLLM导入失败: {e}")
            return False
        
        # 2. 检查AsyncLLMEngine的方法
        logger.info("2. 检查AsyncLLMEngine的可用方法...")
        engine_methods = [method for method in dir(AsyncLLMEngine) if not method.startswith('_')]
        logger.info(f"AsyncLLMEngine可用方法: {engine_methods}")
        
        # 3. 查找abort相关方法
        abort_related = [method for method in engine_methods if any(keyword in method.lower() for keyword in ['abort', 'cancel', 'stop', 'terminate'])]
        logger.info(f"Abort相关方法: {abort_related}")
        
        # 4. 检查方法签名
        logger.info("3. 检查关键方法的签名...")
        import inspect
        
        if hasattr(AsyncLLMEngine, 'generate'):
            sig = inspect.signature(AsyncLLMEngine.generate)
            logger.info(f"generate方法签名: {sig}")
        
        for method_name in abort_related:
            if hasattr(AsyncLLMEngine, method_name):
                try:
                    sig = inspect.signature(getattr(AsyncLLMEngine, method_name))
                    logger.info(f"{method_name}方法签名: {sig}")
                except Exception as e:
                    logger.warning(f"获取{method_name}签名失败: {e}")
        
        # 5. 尝试创建引擎参数（不实际创建引擎）
        logger.info("4. 测试创建引擎参数...")
        try:
            engine_args = AsyncEngineArgs(
                model="gpt2",  # 使用小模型
                max_num_seqs=2,
                gpu_memory_utilization=0.1,
                tensor_parallel_size=1
            )
            logger.info("✓ 引擎参数创建成功")
            
            # 检查引擎参数的属性
            args_attrs = [attr for attr in dir(engine_args) if not attr.startswith('_')]
            logger.info(f"引擎参数可用属性: {args_attrs}")
            
        except Exception as e:
            logger.error(f"❌ 引擎参数创建失败: {e}")
            return False
        
        # 6. 检查SamplingParams
        logger.info("5. 测试SamplingParams...")
        try:
            sampling_params = SamplingParams(
                max_tokens=10,
                temperature=0.7
            )
            logger.info("✓ SamplingParams创建成功")
            
            params_attrs = [attr for attr in dir(sampling_params) if not attr.startswith('_')]
            logger.info(f"SamplingParams可用属性: {params_attrs}")
            
        except Exception as e:
            logger.error(f"❌ SamplingParams创建失败: {e}")
            return False
        
        logger.info("=== API可用性测试完成 ===")
        return True
        
    except Exception as e:
        logger.error(f"❌ 测试过程中出现未预期错误: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_vllm_engine_creation():
    """尝试实际创建引擎（如果环境允许）"""
    try:
        logger.info("=== 尝试创建vLLM引擎 ===")
        
        from vllm import AsyncLLMEngine, AsyncEngineArgs
        
        # 使用最小配置
        engine_args = AsyncEngineArgs(
            model="gpt2",
            max_num_seqs=1,
            gpu_memory_utilization=0.1,
            tensor_parallel_size=1,
            trust_remote_code=True,
            enforce_eager=True,  # 强制使用eager模式，避免编译
            disable_log_stats=True,  # 禁用统计日志
        )
        
        logger.info("正在创建AsyncLLMEngine（这可能需要一些时间）...")
        
        # 设置超时，避免长时间等待
        try:
            engine = await asyncio.wait_for(
                AsyncLLMEngine.from_engine_args(engine_args), 
                timeout=60.0
            )
            logger.info("✓ AsyncLLMEngine创建成功！")
            
            # 检查引擎的实际方法
            logger.info("检查引擎实例的方法...")
            instance_methods = [method for method in dir(engine) if not method.startswith('_')]
            logger.info(f"引擎实例方法: {instance_methods}")
            
            # 查找abort相关方法
            abort_methods = [method for method in instance_methods if any(keyword in method.lower() for keyword in ['abort', 'cancel', 'stop', 'terminate'])]
            logger.info(f"引擎实例Abort相关方法: {abort_methods}")
            
            # 检查内部引擎
            if hasattr(engine, 'engine'):
                logger.info("检查内部引擎...")
                inner_methods = [method for method in dir(engine.engine) if not method.startswith('_')]
                inner_abort_methods = [method for method in inner_methods if any(keyword in method.lower() for keyword in ['abort', 'cancel', 'stop', 'terminate'])]
                logger.info(f"内部引擎Abort相关方法: {inner_abort_methods}")
            
            return engine
            
        except asyncio.TimeoutError:
            logger.error("❌ 引擎创建超时（60秒）")
            return None
        
    except Exception as e:
        logger.error(f"❌ 引擎创建失败: {e}")
        import traceback
        traceback.print_exc()
        return None


async def main():
    """主测试函数"""
    logger.info("开始vLLM简化测试...")
    
    # 测试1: API可用性
    api_available = await test_vllm_api_availability()
    
    if not api_available:
        logger.error("❌ API不可用，跳过后续测试")
        return
    
    # 测试2: 尝试创建引擎
    engine = await test_vllm_engine_creation()
    
    if engine:
        logger.info("✓ 所有测试通过！vLLM可以正常使用")
        
        # 尝试清理引擎
        try:
            if hasattr(engine, 'shutdown'):
                await engine.shutdown()
            elif hasattr(engine, 'engine') and hasattr(engine.engine, 'shutdown'):
                engine.engine.shutdown()
        except Exception as e:
            logger.warning(f"清理引擎时出现警告: {e}")
    else:
        logger.warning("⚠️ 引擎创建失败，但API可用")
    
    logger.info("测试完成")


if __name__ == "__main__":
    asyncio.run(main()) 
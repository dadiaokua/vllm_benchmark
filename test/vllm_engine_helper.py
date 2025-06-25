#!/usr/bin/env python3
"""
vLLM引擎启动和管理工具
"""

import asyncio
import logging
import os
from typing import Optional
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

# 设置环境变量减少c10d网络警告
os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
os.environ.setdefault("NCCL_DEBUG", "WARN")
# 禁用Ray的一些警告和自动启动
os.environ.setdefault("RAY_DISABLE_IMPORT_WARNING", "1")
os.environ.setdefault("RAY_LOG_TO_STDERR", "0")

logger = logging.getLogger(__name__)


def setup_vllm_logging(log_level: str = "WARNING", suppress_engine_logs: bool = True):
    """
    设置vLLM相关的日志级别
    
    Args:
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        suppress_engine_logs: 是否抑制引擎请求/完成相关的详细日志
    """
    # 设置vLLM相关的logger
    vllm_loggers = [
        'vllm',
        'vllm.async_llm_engine',
        'vllm.engine.async_llm_engine',
        'vllm.core.scheduler',
        'vllm.worker.model_runner',
        'vllm.executor.gpu_executor',
        'vllm.model_executor.model_loader'
    ]
    
    log_level_num = getattr(logging, log_level.upper(), logging.WARNING)
    
    for logger_name in vllm_loggers:
        vllm_logger = logging.getLogger(logger_name)
        vllm_logger.setLevel(log_level_num)
        
        # 如果要抑制引擎日志，特别处理async_llm_engine
        if suppress_engine_logs and 'async_llm_engine' in logger_name:
            # 添加一个过滤器来屏蔽特定消息
            class EngineLogFilter(logging.Filter):
                def filter(self, record):
                    # 屏蔽"Added request"和"Finished request"消息
                    message = record.getMessage()
                    if ("Added request" in message or 
                        "Finished request" in message or
                        "Aborted request" in message):
                        return False
                    return True
            
            vllm_logger.addFilter(EngineLogFilter())
    
    logger.info(f"✓ vLLM日志级别设置为: {log_level}, 抑制引擎日志: {suppress_engine_logs}")


class VLLMEngineManager:
    """vLLM引擎管理器"""

    def __init__(self):
        self.engine: Optional[AsyncLLMEngine] = None

    async def start_engine(self,
                           model_path: str = "/home/llm/model_hub/Llama-3.1-8B",
                           max_num_seqs: int = 4,
                           tensor_parallel_size: int = 8,
                           enable_chunked_prefill: bool = False,
                           dtype: str = 'float16',
                           scheduling_policy: str = "priority",
                           log_level: str = "WARNING",
                           suppress_engine_logs: bool = True) -> AsyncLLMEngine:
        """启动vLLM引擎"""
        
        # 设置日志
        setup_vllm_logging(log_level, suppress_engine_logs)
        
        logger.info(f"正在启动vLLM引擎，模型: {model_path}")
        logger.info(f"配置: tensor_parallel_size={tensor_parallel_size}, scheduling_policy={scheduling_policy}")

        engine_args = AsyncEngineArgs(
            model=model_path,
            max_num_seqs=max_num_seqs,
            tensor_parallel_size=tensor_parallel_size,
            enable_chunked_prefill=enable_chunked_prefill,
            dtype=dtype,
            scheduling_policy=scheduling_policy,
            max_model_len=8124,
            enable_prefix_caching=False,
        )

        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        logger.info("✓ vLLM引擎启动成功")
        return self.engine

    async def shutdown_engine(self):
        """关闭引擎"""
        if self.engine:
            try:
                if hasattr(self.engine, 'shutdown'):
                    await self.engine.shutdown()
                    logger.info("✓ 引擎清理完成")
            except Exception as e:
                logger.warning(f"引擎清理时出现警告: {e}")
            finally:
                self.engine = None

    async def create_engine(self, 
                           model_path: str = "/home/llm/model_hub/Llama-3.1-8B",
                           max_num_seqs: int = 4,
                           tensor_parallel_size: int = 8,
                           suppress_logs: bool = True) -> AsyncLLMEngine:
        """
        创建vLLM引擎的便捷方法
        
        Args:
            model_path: 模型路径
            max_num_seqs: 最大并发序列数
            tensor_parallel_size: 张量并行大小
            suppress_logs: 是否抑制详细日志 (默认True)
        
        Returns:
            AsyncLLMEngine实例
        """
        log_level = "WARNING" if suppress_logs else "INFO"
        return await self.start_engine(
            model_path=model_path,
            max_num_seqs=max_num_seqs,
            tensor_parallel_size=tensor_parallel_size,
            log_level=log_level,
            suppress_engine_logs=suppress_logs
        )
    
    async def cleanup(self):
        """清理引擎资源 (shutdown_engine的别名)"""
        await self.shutdown_engine()

    def abort_request(self, request_id: str) -> bool:
        """取消指定请求"""
        if not self.engine:
            logger.error("引擎未启动")
            return False

        try:
            if hasattr(self.engine, 'abort'):
                result = self.engine.abort(request_id)
                logger.info(f"✓ abort请求成功: {request_id}")
                return True
            elif hasattr(self.engine, 'engine') and hasattr(self.engine.engine, 'abort_request'):
                result = self.engine.engine.abort_request(request_id)
                logger.info(f"✓ abort_request请求成功: {request_id}")
                return True
            else:
                logger.error("❌ 引擎不支持abort功能")
                return False
        except Exception as e:
            logger.error(f"❌ abort请求失败: {e}")
            return False


def create_sampling_params(max_tokens: int = 500,
                           temperature: float = 0.7,
                           top_p: float = 0.9) -> SamplingParams:
    """创建采样参数"""
    return SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p
    )


def quick_suppress_vllm_logs():
    """
    快速抑制vLLM的详细日志输出
    这是一个便捷函数，可以在任何地方调用来立即抑制vLLM日志
    """
    setup_vllm_logging("WARNING", True)
    print("✓ vLLM详细日志已抑制")


def restore_vllm_logs():
    """
    恢复vLLM的详细日志输出
    """
    setup_vllm_logging("INFO", False)
    print("✓ vLLM日志已恢复")

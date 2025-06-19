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
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
# 禁用Ray的一些警告和自动启动
os.environ.setdefault("RAY_DISABLE_IMPORT_WARNING", "1")
os.environ.setdefault("RAY_LOG_TO_STDERR", "0")

logger = logging.getLogger(__name__)


class VLLMEngineManager:
    """vLLM引擎管理器"""

    def __init__(self):
        self.engine: Optional[AsyncLLMEngine] = None

    async def start_engine(self,
                           model_path: str = "/home/llm/model_hub/Llama-3.1-8B",
                           max_num_seqs: int = 4,
                           gpu_memory_utilization: float = 0.9,
                           tensor_parallel_size: int = 4,
                           trust_remote_code: bool = True,
                           enable_chunked_prefill: bool = False,
                           dtype: str = 'float16',
                           scheduling_policy: str = "fcfs") -> AsyncLLMEngine:
        """启动vLLM引擎"""
        logger.info(f"正在启动vLLM引擎，模型: {model_path}")
        logger.info(f"配置: tensor_parallel_size={tensor_parallel_size}, scheduling_policy={scheduling_policy}")

        engine_args = AsyncEngineArgs(
            model=model_path,
            max_num_seqs=max_num_seqs,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=trust_remote_code,
            enable_chunked_prefill=enable_chunked_prefill,
            dtype=dtype,
            scheduling_policy=scheduling_policy,
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

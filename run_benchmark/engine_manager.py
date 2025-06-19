#!/usr/bin/env python3
"""
vLLM引擎管理模块
处理vLLM引擎的启动、停止和配置
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


async def start_vllm_engine(args, logger) -> Optional[object]:
    """
    启动vLLM引擎
    
    Args:
        args: 解析后的命令行参数
        logger: 日志记录器
        
    Returns:
        engine: AsyncLLMEngine对象，如果启动失败则返回None
    """
    try:
        from vllm import AsyncLLMEngine, AsyncEngineArgs

        # 只从args中读取vLLM启动参数，如果没有则使用默认值
        model_path = getattr(args, 'model_path', "/home/llm/model_hub/Qwen2.5-32B-Instruct")
        if not model_path:
            logger.error("Model path is required. Please specify --model_path")
            return None

        tensor_parallel_size = getattr(args, 'tensor_parallel_size', 8)
        pipeline_parallel_size = getattr(args, 'pipeline_parallel_size', 1)
        gpu_memory_utilization = getattr(args, 'gpu_memory_utilization', 0.9)
        max_model_len = getattr(args, 'max_model_len', 8124)
        max_num_seqs = getattr(args, 'max_num_seqs', 256)
        max_num_batched_tokens = getattr(args, 'max_num_batched_tokens', 65536)
        swap_space = getattr(args, 'swap_space', 4)
        device = getattr(args, 'device', "cuda")
        dtype = getattr(args, 'dtype', "float16")
        quantization = getattr(args, 'quantization', "None")
        trust_remote_code = getattr(args, 'trust_remote_code', True)
        enable_chunked_prefill = getattr(args, 'enable_chunked_prefill', False)
        disable_log_stats = getattr(args, 'disable_log_stats', False)
        scheduling_policy = getattr(args, 'scheduling_policy', "priority")

        logger.info("Starting vLLM engine with AsyncLLMEngine...")
        logger.info(f"Model path: {model_path}")
        logger.info(f"Tensor parallel size: {tensor_parallel_size}")
        logger.info(f"Pipeline parallel size: {pipeline_parallel_size}")
        logger.info(f"GPU memory utilization: {gpu_memory_utilization}")
        logger.info(f"Max model length: {max_model_len}")
        logger.info(f"Max sequences: {max_num_seqs}")
        logger.info(f"Max batched tokens: {max_num_batched_tokens}")
        logger.info(f"Device: {device}")
        logger.info(f"Data type: {dtype}")
        logger.info(f"Quantization: {quantization}")
        logger.info(f"Trust remote code: {trust_remote_code}")
        logger.info(f"Swap space: {swap_space}GB")
        logger.info(f"Scheduling policy: {scheduling_policy}")

        # 构建引擎参数
        engine_args_dict = {
            "model": model_path,
            "tokenizer": model_path,  # 通常tokenizer和model路径相同
            "tensor_parallel_size": tensor_parallel_size,
            "pipeline_parallel_size": pipeline_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_num_seqs": max_num_seqs,
            "max_num_batched_tokens": max_num_batched_tokens,
            "swap_space": swap_space,
            "device": device,
            "dtype": dtype,
            "trust_remote_code": trust_remote_code,
            "enable_chunked_prefill": enable_chunked_prefill,
            "disable_log_stats": disable_log_stats,
            "scheduling_policy": scheduling_policy
        }

        # 添加可选参数
        if max_model_len:
            engine_args_dict["max_model_len"] = max_model_len

        # 处理量化参数
        if quantization and quantization.lower() != "none":
            engine_args_dict["quantization"] = quantization

        logger.info(f"Engine arguments: {engine_args_dict}")

        # 创建引擎参数对象
        engine_args = AsyncEngineArgs(**engine_args_dict)

        logger.info("Creating AsyncLLMEngine...")

        # 创建异步引擎
        engine = AsyncLLMEngine.from_engine_args(engine_args)

        logger.info("vLLM AsyncLLMEngine started successfully!")

        return engine

    except ImportError as e:
        logger.error(f"Failed to import vLLM: {e}")
        logger.error("Please install vLLM: pip install vllm")
        return None
    except Exception as e:
        logger.error(f"Failed to start vLLM engine: {e}")
        return None


def stop_vllm_engine(engine, logger):
    """
    停止vLLM引擎
    
    Args:
        engine: AsyncLLMEngine对象
        logger: 日志记录器
    """
    if engine is None:
        return

    try:
        logger.info("Stopping vLLM AsyncLLMEngine...")

        # AsyncLLMEngine 通常会在程序结束时自动清理
        # 如果有特定的清理方法，可以在这里调用
        if hasattr(engine, 'engine') and hasattr(engine.engine, 'stop'):
            engine.engine.stop()

        logger.info("vLLM AsyncLLMEngine stopped")

    except Exception as e:
        logger.error(f"Error stopping vLLM engine: {e}") 
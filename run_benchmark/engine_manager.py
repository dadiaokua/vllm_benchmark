#!/usr/bin/env python3
"""
引擎管理模块
处理vLLM引擎的启动、停止和配置
"""

import asyncio
import logging
import os
import subprocess
import time
import signal
import psutil

# 导入vLLM相关模块
try:
    from vllm import AsyncLLMEngine
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncEngineDeadError
    vllm_available = True
except ImportError:
    vllm_available = False

logger = logging.getLogger(__name__)

# 全局变量存储引擎进程
vllm_process = None


def get_gpu_count():
    """获取可用的GPU数量"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=count', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, check=True)
        return int(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        logger.warning("无法获取GPU数量，假设使用CPU")
        return 0


def adjust_engine_config_for_resources(args):
    """根据可用资源调整引擎配置"""
    gpu_count = get_gpu_count()
    
    if gpu_count == 0:
        logger.warning("未检测到GPU，将使用CPU模式（性能会显著降低）")
        args.tensor_parallel_size = 1
        args.gpu_memory_utilization = 0.5
    elif gpu_count < args.tensor_parallel_size:
        logger.warning(f"可用GPU数量({gpu_count})少于tensor_parallel_size({args.tensor_parallel_size})，自动调整")
        args.tensor_parallel_size = gpu_count
    
    # 保守的资源配置
    args.max_num_seqs = min(getattr(args, 'max_num_seqs', 128), 1)
    args.gpu_memory_utilization = min(args.gpu_memory_utilization, 0.8)
    
    logger.info(f"调整后的引擎配置: tensor_parallel_size={args.tensor_parallel_size}, "
                f"gpu_memory_utilization={args.gpu_memory_utilization}, max_num_seqs={args.max_num_seqs}")


async def start_vllm_engine(args, logger):
    """启动vLLM引擎"""
    if not vllm_available:
        logger.error("vLLM not available, cannot start engine")
        return None
    
    try:
        # 调整配置以匹配可用资源
        adjust_engine_config_for_resources(args)
        
        # 设置环境变量以减少警告
        os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
        os.environ.setdefault("RAY_DISABLE_IMPORT_WARNING", "1")
        
        # 从args获取引擎参数，如果没有则使用默认值
        engine_args = AsyncEngineArgs(
            model=getattr(args, 'model_path', '/path/to/model'),
            tensor_parallel_size=getattr(args, 'tensor_parallel_size', 1),
            gpu_memory_utilization=getattr(args, 'gpu_memory_utilization', 0.8),
            max_num_seqs=getattr(args, 'max_num_seqs', 1),
            max_model_len=getattr(args, 'max_model_len', 4096),
            trust_remote_code=getattr(args, 'trust_remote_code', True),
            disable_log_stats=getattr(args, 'disable_log_stats', True),
            enable_prefix_caching=getattr(args, 'enable_prefix_caching', False),
            swap_space=getattr(args, 'swap_space', 0),
            dtype=getattr(args, 'dtype', 'auto'),
            quantization=getattr(args, 'quantization', None) if getattr(args, 'quantization', 'None') != 'None' else None,
        )
        
        logger.info("Creating AsyncLLMEngine with args:")
        logger.info(f"  model: {engine_args.model}")
        logger.info(f"  tensor_parallel_size: {engine_args.tensor_parallel_size}")
        logger.info(f"  gpu_memory_utilization: {engine_args.gpu_memory_utilization}")
        logger.info(f"  max_num_seqs: {engine_args.max_num_seqs}")
        logger.info(f"  max_model_len: {engine_args.max_model_len}")
        logger.info(f"  quantization: {engine_args.quantization}")
        
        # 创建引擎实例
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        
        # 测试引擎是否正常工作
        await asyncio.sleep(2)  # 给引擎一些初始化时间
        
        logger.info("vLLM AsyncLLMEngine started successfully!")
        return engine
        
    except Exception as e:
        logger.error(f"Failed to start vLLM engine: {e}")
        import traceback
        traceback.print_exc()
        return None


def stop_vllm_engine(engine, logger):
    """停止vLLM引擎"""
    global vllm_process
    
    if engine:
        try:
            # 如果引擎有stop方法，调用它
            if hasattr(engine, 'stop'):
                engine.stop()
            logger.info("vLLM engine stopped")
        except Exception as e:
            logger.warning(f"Error stopping vLLM engine: {e}")
    
    # 清理可能存在的进程
    if vllm_process and vllm_process.poll() is None:
        try:
            # 先尝试优雅关闭
            vllm_process.terminate()
            vllm_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            # 如果优雅关闭失败，强制杀死
            vllm_process.kill()
            vllm_process.wait()
        except Exception as e:
            logger.warning(f"Error cleaning up vLLM process: {e}")
        finally:
            vllm_process = None


def cleanup_vllm_processes():
    """清理所有vLLM相关进程"""
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'vllm' in proc.info['name'].lower() or \
                   any('vllm' in arg.lower() for arg in proc.info['cmdline'] if arg):
                    proc.kill()
                    logger.info(f"Killed vLLM process: {proc.info['pid']}")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except Exception as e:
        logger.warning(f"Error during vLLM process cleanup: {e}") 
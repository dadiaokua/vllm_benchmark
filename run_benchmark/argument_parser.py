#!/usr/bin/env python3
"""
参数解析和预处理模块
处理命令行参数的解析、验证和预处理
"""

import argparse
import logging


def safe_float_conversion(value, default=1.0):
    """安全地将字符串转换为float，处理空字符串和无效值"""
    if not value or value.strip() == '':
        return default
    try:
        return float(value.strip())
    except (ValueError, TypeError):
        return default


def preprocess_space_separated_args(args):
    """预处理空格分隔的参数，将单个字符串拆分为列表"""

    # 处理 short_qpm - 只在是单个包含空格的字符串时才拆分
    if args.short_qpm and len(args.short_qpm) == 1 and ' ' in str(args.short_qpm[0]):
        args.short_qpm = str(args.short_qpm[0]).split()

    # 处理 long_qpm - 只在是单个包含空格的字符串时才拆分
    if args.long_qpm and len(args.long_qpm) == 1 and ' ' in str(args.long_qpm[0]):
        args.long_qpm = str(args.long_qpm[0]).split()

    # 处理 short_clients_slo - 只在是单个包含空格的字符串时才拆分
    if args.short_clients_slo and len(args.short_clients_slo) == 1 and ' ' in str(args.short_clients_slo[0]):
        args.short_clients_slo = str(args.short_clients_slo[0]).split()

    # 处理 long_clients_slo - 只在是单个包含空格的字符串时才拆分
    if args.long_clients_slo and len(args.long_clients_slo) == 1 and ' ' in str(args.long_clients_slo[0]):
        args.long_clients_slo = str(args.long_clients_slo[0]).split()

    return args


def parse_args(logger):
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Run vLLM benchmarks with various configurations")
    
    # 基础连接参数
    parser.add_argument("--vllm_url", type=str, nargs='+', required=True,
                        help="URLs of the vLLM servers (can provide multiple)",
                        default=["http://127.0.0.1"])
    parser.add_argument("--use_tunnel", type=int, default=1)
    parser.add_argument("--api_key", type=str, required=True, help="API key for vLLM server", default='test')
    parser.add_argument("--local_port", type=int, nargs='+', required=True, help="local port", default=[8080])
    parser.add_argument("--remote_port", type=int, nargs='+', required=True, help="remote ssh port", default=[8080])
    
    # 请求配置参数
    parser.add_argument("--distribution", type=str, help="Distribution of request")
    parser.add_argument("--short_qpm", type=str, nargs='+', help="Qps of short client request", required=True,
                        default=1.0)
    parser.add_argument("--short_client_qpm_ratio", type=float, required=True, help="Qps ratio of short client",
                        default=1)
    parser.add_argument("--long_qpm", type=str, nargs='+', help="Qps of long client request", required=True,
                        default=1.0)
    parser.add_argument("--long_client_qpm_ratio", type=float, required=True, help="Qps ratio of long client",
                        default=1)
    
    # 客户端配置参数
    parser.add_argument("--short_clients", type=int, help="Number of client send short context", default=1)
    parser.add_argument("--short_clients_slo", type=str, nargs='+', required=True, help="Slo of short client")
    parser.add_argument("--long_clients", type=int, help="Number of client send long context", default=1)
    parser.add_argument("--long_clients_slo", type=str, nargs='+', required=True, help="Slo of long client")
    
    # 并发和性能参数
    parser.add_argument("--concurrency", type=int, help="concurrency", default=50)
    parser.add_argument("--num_requests", type=int, help="Number of requests", default=1000)
    parser.add_argument("--request_timeout", type=int, default=5,
                        help="Timeout for each request in seconds (default: 30)")
    parser.add_argument("--sleep", type=int, help="Sleep time per concurrency", default=60)
    
    # 实验配置参数
    parser.add_argument("--round", type=int, default=20, help="Round of Exp.", required=True)
    parser.add_argument("--round_time", type=int, default=600, help="Timeout for every round (default: 600)",
                        required=True)
    parser.add_argument("--exp", type=str, help="Experiment type", required=True, default="LFS")
    parser.add_argument("--use_time_data", type=int, help="whether use time data", default=0)
    
    # 模型和tokenizer参数
    parser.add_argument("--tokenizer", type=str, help="Tokenizer local path",
                        default="/Users/myrick/modelHub/hub/Meta-Llama-3.1-8B-Instruct-AWQ-INT4")
    parser.add_argument("--request_model_name", type=str, help="Request model name",
                        default="Meta-Llama-3.1-8B-Instruct-AWQ-INT4", required=True)
    
    # vLLM引擎参数
    parser.add_argument("--start_engine", type=bool, help="Whether to start the vLLM engine", default=False, required=True)
    parser.add_argument("--model_path", type=str, help="Path to the vLLM model",
                        default="/home/llm/model_hub/Qwen2.5-32B-Instruct")
    parser.add_argument("--tensor_parallel_size", type=int, help="Tensor parallel size", default=8)
    parser.add_argument("--pipeline_parallel_size", type=int, help="Pipeline parallel size", default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, help="GPU memory utilization", default=0.9)
    parser.add_argument("--max_model_len", type=int, help="Maximum model length", default=8124)
    parser.add_argument("--max_num_seqs", type=int, help="Maximum number of sequences", default=256)
    parser.add_argument("--max_num_batched_tokens", type=int, help="Maximum number of batched tokens", default=65536)
    parser.add_argument("--swap_space", type=int, help="Swap space size in GB", default=4)
    parser.add_argument("--device", type=str, help="Device type", default="cuda")
    parser.add_argument("--dtype", type=str, help="Data type", default="float16")
    parser.add_argument("--quantization", type=str, help="Quantization method", default="None")
    parser.add_argument("--trust_remote_code", action="store_true", help="Trust remote code", default=True)
    parser.add_argument("--enable_chunked_prefill", action="store_true", help="Enable chunked prefill", default=False)
    parser.add_argument("--disable_log_stats", action="store_true", help="Disable log statistics")
    parser.add_argument("--scheduling_policy", action="store_true", help="Log statistics", default="priority")
    
    args = parser.parse_args()
    return args


def print_benchmark_config(args, logger):
    """打印基准测试配置信息"""
    logger.info("\nBenchmark Configuration:")
    logger.info("------------------------")
    for k, v in vars(args).items():
        logger.info(f"{k}: {v}")
    logger.info("------------------------\n")


def validate_args(args, logger):
    """验证参数的合法性"""
    # 预处理参数
    args = preprocess_space_separated_args(args)
    
    # 打印调试信息
    logger.info(f"Processed short_qpm: {args.short_qpm}")
    logger.info(f"Processed long_qpm: {args.long_qpm}")
    logger.info(f"Processed short_clients_slo: {args.short_clients_slo}")
    logger.info(f"Processed long_clients_slo: {args.long_clients_slo}")

    # 验证QPS参数
    if len(args.short_qpm) != 1 and len(args.short_qpm) != args.short_clients:
        logger.error("short_qps must be a single value or a list of values equal to the number of short clients")
        return None

    if len(args.long_qpm) != 1 and len(args.long_qpm) != args.long_clients:
        logger.error("long_qps must be a single value or a list of values equal to the number of long clients")
        return None
    
    # 验证vLLM URL
    if not args.vllm_url or not args.vllm_url[0]:
        logger.error("vLLM URL is required")
        return None
        
    return args 
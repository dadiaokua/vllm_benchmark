import asyncio
import json
import os
import time
import argparse
from datetime import datetime
from transformers import AutoTokenizer
from BenchmarkClient.BenchmarkClient import BenchmarkClient
from BenchmarkMonitor.BenchmarkMonitor import RESULTS_FILE, ExperimentMonitor
from config.Config import GLOBAL_CONFIG
from plot.plotMain import plot_result
from util.BaseUtil import initialize_clients
from util.FileSaveUtil import save_benchmark_results
from util.TunnelUtil import setup_vllm_servers, stop_tunnel
from RequestQueueManager.RequestQueueManager import RequestQueueManager, QueueStrategy
import logging
import subprocess


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


async def setup_benchmark_tasks(args, all_results, request_queue, logger):
    """Setup and create benchmark tasks"""
    tasks = []
    clients = []

    # 预处理参数
    args = preprocess_space_separated_args(args)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    # short_formatted_json, time_data = await prepare_benchmark_data('short', tokenizer)
    # long_formatted_json, time_data = await prepare_benchmark_data('long', tokenizer)

    with open("prompt_hub/short_prompts.json", "r", encoding="utf-8") as f:
        short_formatted_json = json.load(f)

    with open("prompt_hub/long_prompts.json", "r", encoding="utf-8") as f:
        long_formatted_json = json.load(f)

    # if args.exp == "DLPM":
    #     short_formatted_json = make_prefix_list(short_formatted_json, tokenizer, 200)
    #     long_formatted_json = make_prefix_list(long_formatted_json, tokenizer, 1000)

    openAI_client = initialize_clients(args.local_port)

    if not args.vllm_url or not args.vllm_url[0]:
        logger.error("vLLM URL is required")
        return None, None, None

    # 创建共享的队列管理器（如果使用队列实验）
    queue_manager = None
    queue_task = None
    if args.exp.startswith("QUEUE_"):
        
        # 根据实验类型选择队列策略
        strategy_map = {
            "QUEUE_FIFO": QueueStrategy.FIFO,
            "QUEUE_PRIORITY": QueueStrategy.PRIORITY,
            "QUEUE_ROUND_ROBIN": QueueStrategy.ROUND_ROBIN,
            "QUEUE_SJF": QueueStrategy.SHORTEST_JOB_FIRST,
            "QUEUE_FAIR": QueueStrategy.FAIR_SHARE
        }
        
        strategy = strategy_map.get(args.exp, QueueStrategy.FIFO)
        queue_manager = RequestQueueManager(strategy=strategy, max_queue_size=20000)
        queue_manager.set_openai_client(openAI_client)
        
        # 启动队列管理器（在后台运行，不需要保存task引用）
        asyncio.create_task(queue_manager.start_processing(num_workers=10))
        logger.info(f"Created queue manager with strategy: {strategy.value}")

    # 打印调试信息
    logger.info(f"Processed short_qpm: {args.short_qpm}")
    logger.info(f"Processed long_qpm: {args.long_qpm}")
    logger.info(f"Processed short_clients_slo: {args.short_clients_slo}")
    logger.info(f"Processed long_clients_slo: {args.long_clients_slo}")

    if len(args.short_qpm) != 1 and len(args.short_qpm) != args.short_clients:
        logger.error("short_qps must be a single value or a list of values equal to the number of short clients")
        return None, None, None

    if len(args.long_qpm) != 1 and len(args.long_qpm) != args.long_clients:
        logger.error("long_qps must be a single value or a list of values equal to the number of long clients")
        return None, None, None

        # Create short request clients
    for index in range(args.short_clients):
        qpm_value = safe_float_conversion(args.short_qpm[0] if len(args.short_qpm) == 1 else args.short_qpm[index])
        slo_value = safe_float_conversion(
            args.short_clients_slo[0] if len(args.short_clients_slo) == 1 else args.short_clients_slo[index], 10)
        logger.info(f"Creating short client {index}: qpm={qpm_value}, slo={slo_value}")
        client = BenchmarkClient(
            client_type='short',
            client_index=index,
            qpm=qpm_value,
            port=args.local_port,
            api_key=args.api_key,
            distribution=args.distribution,
            request_timeout=args.request_timeout,
            concurrency=args.concurrency,
            round_time=args.round_time,
            sleep=args.sleep,
            result_queue=all_results,
            use_time_data=args.use_time_data,
            formatted_json=short_formatted_json,
            OpenAI_client=openAI_client,
            tokenizer=tokenizer,
            time_data=None,
            round=args.round,
            exp_type=args.exp,
            qpm_ratio=args.short_client_qpm_ratio,
            latency_slo=int(slo_value),
            queue_manager=queue_manager  # 传递队列管理器
        )
        clients.append(client)
        tasks.append(client.start())

    # Create long request clients
    for index in range(args.long_clients):
        qpm_value = safe_float_conversion(args.long_qpm[0] if len(args.long_qpm) == 1 else args.long_qpm[index])
        slo_value = safe_float_conversion(
            args.long_clients_slo[0] if len(args.long_clients_slo) == 1 else args.long_clients_slo[index], 10)

        client = BenchmarkClient(
            client_type='long',
            client_index=index,
            qpm=qpm_value,
            port=args.local_port,
            api_key=args.api_key,
            distribution=args.distribution,
            request_timeout=args.request_timeout,
            concurrency=args.concurrency,
            round_time=args.round_time,
            sleep=args.sleep,
            result_queue=all_results,
            use_time_data=args.use_time_data,
            formatted_json=long_formatted_json,
            OpenAI_client=openAI_client,
            tokenizer=tokenizer,
            time_data=None,
            round=args.round,
            exp_type=args.exp,
            qpm_ratio=args.long_client_qpm_ratio,
            latency_slo=int(slo_value),
            queue_manager=queue_manager  # 传递队列管理器
        )
        clients.append(client)
        tasks.append(client.start())

    # 创建监控器实例
    monitor = ExperimentMonitor(clients, all_results, args.short_clients + args.long_clients, args.exp, request_queue,
                                args.use_tunnel)

    # 创建监控任务
    monitor_task = asyncio.create_task(monitor())
    tasks.insert(0, monitor_task)

    # 如果使用队列管理器，启动队列处理（但不加入tasks，让它在后台运行）
    if queue_manager:
        # 队列管理器已经在setup_benchmark_tasks中启动了，这里只需要记录一下
        logger.info(f"Queue manager is running in background with strategy: {queue_manager.strategy.value}")

    return tasks, monitor_task, clients, queue_manager


def setup_logger():
    # 日志文件夹和文件名
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "run_benchmarks.log")

    # 设置全局logger
    logger = logging.getLogger("run_benchmarks")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_file, encoding="utf-8", mode="w")
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        # 控制台输出
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


def parse_args(logger):
    parser = argparse.ArgumentParser(description="Run vLLM benchmarks with various configurations")
    parser.add_argument("--vllm_url", type=str, nargs='+', required=True,
                        help="URLs of the vLLM servers (can provide multiple)",
                        default=["http://127.0.0.1"])
    parser.add_argument("--use_tunnel", type=int, default=1)
    parser.add_argument("--api_key", type=str, required=True, help="API key for vLLM server", default='test')
    parser.add_argument("--distribution", type=str, help="Distribution of request")
    parser.add_argument("--short_qpm", type=str, nargs='+', help="Qps of short client request", required=True,
                        default=1.0)
    parser.add_argument("--short_client_qpm_ratio", type=float, required=True, help="Qps ratio of short client",
                        default=1)
    parser.add_argument("--long_qpm", type=str, nargs='+', help="Qps of long client request", required=True,
                        default=1.0)
    parser.add_argument("--long_client_qpm_ratio", type=float, required=True, help="Qps ratio of long client",
                        default=1)
    parser.add_argument("--concurrency", type=int, help="concurrency", default=50)
    parser.add_argument("--num_requests", type=int, help="Number of requests", default=1000)
    parser.add_argument("--short_clients", type=int, help="Number of client send short context", default=1)
    parser.add_argument("--short_clients_slo", type=str, nargs='+', required=True, help="Slo of short client")
    parser.add_argument("--long_clients", type=int, help="Number of client send long context", default=1)
    parser.add_argument("--long_clients_slo", type=str, nargs='+', required=True, help="Slo of long client")
    parser.add_argument("--sleep", type=int, help="Sleep time per concurrency", default=60)
    parser.add_argument("--local_port", type=int, nargs='+', required=True, help="local port", default=[8080])
    parser.add_argument("--remote_port", type=int, nargs='+', required=True, help="remote ssh port", default=[8080])
    parser.add_argument("--use_time_data", type=int, help="whether use time data", default=0)
    parser.add_argument("--request_timeout", type=int, default=5,
                        help="Timeout for each request in seconds (default: 30)")
    parser.add_argument("--round", type=int, default=20, help="Round of Exp.", required=True)
    parser.add_argument("--round_time", type=int, default=600, help="Timeout for every round (default: 600)",
                        required=True)
    parser.add_argument("--exp", type=str, help="Experiment type", required=True, default="LFS")
    parser.add_argument("--tokenizer", type=str, help="Tokenizer local path",
                        default="/Users/myrick/modelHub/hub/Meta-Llama-3.1-8B-Instruct-AWQ-INT4")
    parser.add_argument("--request_model_name", type=str, help="Request model name",
                        default="Meta-Llama-3.1-8B-Instruct-AWQ-INT4", required=True)
    parser.add_argument("--start_engine", type=bool, help="Whether to start the vLLM engine", default=True)
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
    parser.add_argument("--enable_chunked_prefill", action="store_true", help="Enable chunked prefill")
    parser.add_argument("--disable_log_stats", action="store_true", help="Disable log statistics")
    args = parser.parse_args()
    return args


def print_benchmark_config(args, logger):
    logger.info("\nBenchmark Configuration:")
    logger.info("------------------------")
    for k, v in vars(args).items():
        logger.info(f"{k}: {v}")
    logger.info("------------------------\n")


def setup_servers_if_needed(args):
    if getattr(args, "use_tunnel", 0):
        return setup_vllm_servers(args.vllm_url, args.local_port, args.remote_port)
    return []


def setup_request_model_name(args):
    if args.request_model_name:
        GLOBAL_CONFIG['request_model_name'] = args.request_model_name


def prepare_results_file():
    with open(RESULTS_FILE, "w") as f:
        json.dump([], f)


async def run_benchmark_tasks(tasks, logger):
    benchmark_timeout = GLOBAL_CONFIG.get('exp_time', 36000)
    try:
        await asyncio.wait_for(asyncio.gather(*tasks[1:]), timeout=benchmark_timeout)
    except asyncio.TimeoutError:
        logger.error(f"Tasks did not complete within {benchmark_timeout} seconds, cancelling...")
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def process_and_save_results(tasks, start_time, args, logger):
    all_benchmark_results = []
    for task in tasks[1:]:
        if task.done() and not task.cancelled():
            try:
                result = task.result()
                if result:
                    all_benchmark_results.append(result)
            except Exception as e:
                logger.warning(f"Task result retrieval failed: {e}")

    benchmark_results = all_benchmark_results
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Total time: {total_time:.2f} seconds")

    start_datetime = datetime.fromtimestamp(start_time)
    end_datetime = datetime.fromtimestamp(end_time)
    filename = (
        f"{args.exp}_{start_datetime.strftime('%m%d_%H-%M')}_to_{end_datetime.strftime('%H-%M')}.json"
    ).replace(" ", "_").replace(":", "-").replace("/", "-")

    args_dict = vars(args)
    plot_data = {
        "filename": filename,
        "total_time": round(total_time, 2),
    }
    plot_data.update(args_dict)
    save_benchmark_results(filename, benchmark_results, plot_data, logger)
    return benchmark_results, total_time, filename, plot_data


async def cancel_monitor_task(monitor_task, logger):
    monitor_task.cancel()
    try:
        await monitor_task
    except asyncio.CancelledError:
        logger.info("Monitor task cancelled.")


async def start_vllm_engine(args, logger):
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
            "disable_log_stats": disable_log_stats
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


async def main():
    
    logger = setup_logger()
    args = parse_args(logger)
    print_benchmark_config(args, logger)
    
    # 启动vLLM引擎
    vllm_engine = None
    if getattr(args, 'start_engine', True):
        vllm_engine = await start_vllm_engine(args, logger)
        if vllm_engine is None:
            logger.error("Failed to start vLLM engine, exiting...")
            return
        # 添加vLLM引擎到全局配置，以便其他模块可以访问
        GLOBAL_CONFIG['vllm_engine'] = vllm_engine
    
    GLOBAL_CONFIG['round_time'] = args.round_time
    if GLOBAL_CONFIG.get('exp_time', 36000) < args.round_time * args.round:
        GLOBAL_CONFIG['exp_time'] = args.round_time * args.round * 3

    servers = setup_servers_if_needed(args)
    setup_request_model_name(args)
    prepare_results_file()

    all_results = asyncio.Queue()
    request_queue = asyncio.Queue()

    start_time = time.time()
    
    try:
        tasks, monitor_task, clients, queue_manager = await setup_benchmark_tasks(args, all_results, request_queue, logger)

        try:
            await run_benchmark_tasks(tasks, logger)
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")

        benchmark_results, total_time, filename, plot_data = process_and_save_results(
            tasks, start_time, args, logger
        )

        for server in servers:
            stop_tunnel(server)

        await cancel_monitor_task(monitor_task, logger)
        
        # 停止队列管理器（如果存在）
        if queue_manager:
            await queue_manager.stop()
            logger.info("Queue manager stopped")
        
        plot_result(plot_data)
        
    finally:
        # 确保在程序结束时停止vLLM引擎
        if vllm_engine:
            stop_vllm_engine(vllm_engine, logger)


if __name__ == "__main__":
    asyncio.run(main())

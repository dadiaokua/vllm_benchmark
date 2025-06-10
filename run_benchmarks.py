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
from util.JsonFormatterUtil import prepare_benchmark_data, make_prefix_list
from util.TunnelUtil import setup_vllm_servers, stop_tunnel
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
            latency_slo=int(slo_value)
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
            latency_slo=int(slo_value)
        )
        clients.append(client)
        tasks.append(client.start())

    # 创建监控器实例
    monitor = ExperimentMonitor(clients, all_results, args.short_clients + args.long_clients, args.exp, request_queue,
                                args.use_tunnel)

    # 创建监控任务
    monitor_task = asyncio.create_task(monitor())
    tasks.insert(0, monitor_task)

    return tasks, monitor_task, clients


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
    benchmark_timeout = GLOBAL_CONFIG.get('exp_time', 3600 * 2)
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
        if task.done():
            result = task.result()
            if result:
                all_benchmark_results.append(result)

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


async def main():
    logger = setup_logger()
    args = parse_args(logger)
    print_benchmark_config(args, logger)
    GLOBAL_CONFIG['round_time'] = args.round_time
    if GLOBAL_CONFIG.get('exp_time', 1) < args.round_time * args.round:
        GLOBAL_CONFIG['exp_time'] = args.round_time * args.round * 1.5

    servers = setup_servers_if_needed(args)
    setup_request_model_name(args)
    prepare_results_file()

    all_results = asyncio.Queue()
    request_queue = asyncio.Queue()

    start_time = time.time()
    tasks, monitor_task, clients = await setup_benchmark_tasks(args, all_results, request_queue, logger)

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
    plot_result(plot_data)


if __name__ == "__main__":
    asyncio.run(main())

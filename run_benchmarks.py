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


async def setup_benchmark_tasks(args, all_results):
    """Setup and create benchmark tasks"""
    tasks = []
    clients = []

    tokenizer = AutoTokenizer.from_pretrained(
        '/Users/myrick/modelHub/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659',
        trust_remote_code=True)

    # short_formatted_json, time_data = await prepare_benchmark_data('short', args.exp, tokenizer, max_request_number)
    # long_formatted_json, time_data = await prepare_benchmark_data('long', args.exp, tokenizer, max_request_number)

    with open("prompt_hub/short_prompts.json", "r", encoding="utf-8") as f:
        short_formatted_json = json.load(f)

    with open("prompt_hub/long_prompts.json", "r", encoding="utf-8") as f:
        long_formatted_json = json.load(f)

    # if args.exp == "DLPM":
    #     short_formatted_json = make_prefix_list(short_formatted_json, tokenizer, 200)
    #     long_formatted_json = make_prefix_list(long_formatted_json, tokenizer, 1000)

    openAI_client = initialize_clients(args.local_port)

    if len(args.short_qps) != 1 and len(args.short_qps) != args.short_clients:
        print("short_qps must be a single value or a list of values equal to the number of short clients")
        return None, None, None
    
    if len(args.long_qps) != 1 and len(args.long_qps) != args.long_clients:
        print("long_qps must be a single value or a list of values equal to the number of long clients")
        return None, None, None 

    # Create short request clients
    for index in range(args.short_clients):
        client = BenchmarkClient(
            client_type='short',
            client_index=index,
            qps=int(args.short_qps[0] if len(args.short_qps) == 1 else args.short_qps[index]),
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
            qps_ratio=args.short_client_qps_ratio,
        )
        clients.append(client)
        tasks.append(client.start())

    # Create long request clients
    for index in range(args.long_clients):
        client = BenchmarkClient(
            client_type='long',
            client_index=index,
            qps=int(args.long_qps[0] if len(args.long_qps) == 1 else args.long_qps[index]),
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
            qps_ratio=args.long_client_qps_ratio
        )
        clients.append(client)
        tasks.append(client.start())

    # 创建监控器实例
    monitor = ExperimentMonitor(clients, all_results, args.short_clients + args.long_clients, args.exp)

    # 创建监控任务
    monitor_task = asyncio.create_task(monitor())
    tasks.insert(0, monitor_task)

    return tasks, monitor_task, clients


async def main():
    parser = argparse.ArgumentParser(description="Run vLLM benchmarks with various configurations")
    parser.add_argument("--vllm_url", type=str, nargs='+', required=True,
                        help="URLs of the vLLM servers (can provide multiple)",
                        default=["http://127.0.0.1"])
    parser.add_argument("--use_tunnel", type=int, default=1)
    parser.add_argument("--api_key", type=str, required=True, help="API key for vLLM server", default='test')
    parser.add_argument("--distribution", type=str, help="Distribution of request")
    parser.add_argument("--short_qps", type=str, nargs='+', help="Qps of short client request", required=True, default=1.0)
    parser.add_argument("--short_client_qps_ratio", type=float, required=True, help="Qps ratio of short client",
                        default=1)
    parser.add_argument("--long_qps", type=str, nargs='+', help="Qps of long client request", required=True, default=1.0)
    parser.add_argument("--long_client_qps_ratio", type=float, required=True, help="Qps ratio of long client",
                        default=1)
    parser.add_argument("--concurrency", type=int, help="concurrency", default=50)
    parser.add_argument("--num_requests", type=int, help="Number of requests", default=1000)
    parser.add_argument("--short_clients", type=int, help="Number of client send short context", default=1)
    parser.add_argument("--long_clients", type=int, help="Number of client send long context", default=1)
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

    args = parser.parse_args()

    print("\nBenchmark Configuration:")
    print("------------------------")
    print(f"vLLM Server URL: {args.vllm_url}")
    print(f"Distribution: {args.distribution}")
    print(f"Short QPS: {args.short_qps}")
    print(f"Short Client QPS Ratio: {args.short_client_qps_ratio}")
    print(f"Long QPS: {args.long_qps}")
    print(f"Long Client QPS Ratio: {args.long_client_qps_ratio}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Number of Requests: {args.num_requests}")
    print(f"Short Clients: {args.short_clients}")
    print(f"Long Clients: {args.long_clients}")
    print(f"Sleep Time: {args.sleep} seconds")
    print(f"Local Port: {args.local_port}")
    print(f"Remote Port: {args.remote_port}")
    print(f"Use Time Data: {args.use_time_data}")
    print(f"Request Timeout: {args.request_timeout} seconds")
    print(f"Round: {args.round}")
    print(f"Round Time: {args.round_time} seconds")
    print(f"Experiment Type: {args.exp}")
    print("------------------------\n")
    GLOBAL_CONFIG['round_time'] = args.round_time
    if args.use_tunnel:
        servers = setup_vllm_servers(args.vllm_url, args.local_port, args.remote_port)
    else:
        servers = []

    with open(RESULTS_FILE, "w") as f:
        json.dump([], f)

    all_results = asyncio.Queue()

    start_time = time.time()
    tasks, monitor_task, clients = await setup_benchmark_tasks(args, all_results)

    try:
        benchmark_timeout = GLOBAL_CONFIG.get('exp_time', 3600 * 2)
        await asyncio.wait_for(asyncio.gather(*tasks[1:]), timeout=benchmark_timeout)

    except asyncio.TimeoutError:
        print(f"Tasks did not complete within {benchmark_timeout} seconds, cancelling...")
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    except Exception as e:
        print(f"An error occurred: {e}")
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    all_benchmark_results = []
    for task in tasks[1:]:
        if task.done():
            result = task.result()
            if result:
                all_benchmark_results.append(result)

    tasks[0].done()  # 取消monitor_task

    benchmark_results = all_benchmark_results

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time: {total_time:.2f} seconds")

    start_datetime = datetime.fromtimestamp(start_time)
    end_datetime = datetime.fromtimestamp(end_time)

    # Create a more descriptive filename with date, time, and benchmark parameters
    filename = (
        f"{args.exp}_{start_datetime.strftime('%m%d_%H-%M')}_to_{end_datetime.strftime('%H-%M')}.json"
    )
    filename = filename.replace(" ", "_").replace(":", "-").replace("/", "-")

    plot_data = {
        "exp_type": args.exp,
        "filename": filename,
        "concurrency": args.concurrency,
        "num_requests": args.num_requests,
        "total_time": round(total_time, 2),
        "short_client_qps_ratio": args.short_client_qps_ratio,
        "long_client_qps_ratio": args.long_client_qps_ratio
    }

    save_benchmark_results(filename, benchmark_results, plot_data)

    for server in servers:
        stop_tunnel(server)

    monitor_task.cancel()
    try:
        await monitor_task
    except asyncio.CancelledError:
        print("Monitor task cancelled.")

    plot_result(args.exp, filename, args.concurrency, round(total_time, 2))


if __name__ == "__main__":
    asyncio.run(main())

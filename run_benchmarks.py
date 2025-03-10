import asyncio
import json
import os
import time
import argparse
import traceback

import numpy as np

from BenchmarkMonitor.BenchmarkMonitor import setup_benchmark_monitoring, _BENCHMARK_MONITOR, monitor_benchmark
from plot.plot import plot_result
from util.util import open_jsonl_file, QAJsonFormatter, setup_vllm_servers, stop_tunnel
from vllm_benchmark import run_benchmark
from openai import AsyncOpenAI
from transformers import AutoTokenizer


async def run_all_benchmarks(local_port, api_key, distribution, configurations, concurrency,
                             client_type, index, sleep):
    all_results = []

    # 创建多个client实例
    clients = []
    if isinstance(local_port, list):
        for port in local_port:
            clients.append(AsyncOpenAI(base_url="http://localhost:" + str(port) + "/v1"))
    else:
        clients.append(AsyncOpenAI(base_url="http://localhost:" + str(local_port) + "/v1"))

    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))

    data_path, jsonl_files = open_jsonl_file(client_type, dataset2prompt)

    formatter = QAJsonFormatter()
    formatted_json = []
    # 创建一个新的列表，用于存储长度小于8192的项
    filtered_json = []
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 或 "true"
    tokenizer = AutoTokenizer.from_pretrained(
        '/Users/myrick/modelHub/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659',
        trust_remote_code=True)
    try:
        str_count = 0
        formatted_json = await formatter.format_qa_json(tokenizer, dataset2prompt, 5000,
                                                        jsonl_files, data_path,
                                                        500000 if client_type == 'long' else 100000, client_type)
        for i, item in enumerate(formatted_json):
            if len(str(item)) < 4000:
                str_count = str_count + 1
                filtered_json.append(item)
        print(f'request count:', str_count)
    except Exception as e:
        print(f"Error: {str(e)}")
        print("详细错误信息:")
        print(traceback.format_exc())

    # sample_content = sample_sharegpt_requests(data_path, 100000)
    if formatted_json is None:
        print("No json formatted")
        return None

    client_index = client_type + "_" + str(index)

    for i, config in enumerate(configurations):
        results = await wrapped_run_benchmark(config, concurrency, 20,
                                              clients, distribution, client_index, formatted_json, i,
                                              tokenizer)
        all_results.append(results)
        if results["successful_requests"] * 100 / results['total_requests'] < 70:
            # 如果成功率小于70%，后续配置都使用当前配置的qps
            current_qps = config['qps'] - 5
            for config_round, remaining_config in enumerate(configurations[i + 1:]):
                remaining_config['qps'] = current_qps
                results = await wrapped_run_benchmark(remaining_config, concurrency, 20,
                                                      clients, distribution, client_index, formatted_json,
                                                      config_round + i + 1, tokenizer)
                all_results.append(results)
                time.sleep(sleep)
            break
        time.sleep(sleep)  # Wait a bit between runs to let the system cool down

    return all_results


async def wrapped_run_benchmark(config, concurrency, request_timeout,
                                clients, distribution, client_index, formatted_json, i, tokenizer):
    # 这里调用原始的 run_benchmark
    results = await run_benchmark(
        config['num_requests'], concurrency, request_timeout, config['output_tokens'],
        clients, distribution, config['qps'], client_index, formatted_json, i,
        tokenizer
    )

    # 解析 client_index 获取客户端类型
    client_type = client_index.split("_")[0]
    client_idx = client_index.split("_")[1]

    # 向监控器添加结果
    await monitor_benchmark(client_type, client_idx, i, results)

    return results


async def main():
    parser = argparse.ArgumentParser(description="Run vLLM benchmarks with various configurations")
    parser.add_argument("--vllm_url", type=str, nargs='+', required=True,
                        help="URLs of the vLLM servers (can provide multiple)",
                        default=["http://127.0.0.1"])
    parser.add_argument("--api_key", type=str, required=True, help="API key for vLLM server", default='test')
    parser.add_argument("--use_long_context", action="store_true",
                        help="Use long context prompt pairs instead of short prompts", default=True)
    parser.add_argument("--distribution", type=str, help="Distribution of request")
    parser.add_argument("--short_qps", type=float, help="Qps of short request", required=True, default=1)
    parser.add_argument("--long_qps", type=float, help="Qps of long request", required=True, default=1)
    parser.add_argument("--range_lower", type=float, help="Lower", default=1)
    parser.add_argument("--range_higher", type=float, help="Higher", default=400)
    parser.add_argument("--concurrency", type=int, help="concurrency", default=50)
    parser.add_argument("--num_requests", type=int, help="Number of requests", default=1000)
    parser.add_argument("--short_clients", type=int, help="Number of client send short context", default=1)
    parser.add_argument("--long_clients", type=int, help="Number of client send long context", default=1)
    parser.add_argument("--sleep", type=int, help="Sleep time per concurrency", default=60)
    parser.add_argument("--local_port", type=int, nargs='+', required=True, help="local port", default=[8080])
    parser.add_argument("--remote_port", type=int, nargs='+', required=True, help="remote ssh port", default=[8080])

    args = parser.parse_args()

    print("\nBenchmark Configuration:")
    print("------------------------")
    print(f"vLLM Server URL: {args.vllm_url}")
    print(f"API Key: {args.api_key}")
    print(f"Use Long Context: {args.use_long_context}")
    print(f"Distribution: {args.distribution}")
    print(f"Short QPS: {args.short_qps}")
    print(f"Long QPS: {args.long_qps}")
    print(f"Range: {args.range_lower} - {args.range_higher}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Number of Requests: {args.num_requests}")
    print(f"Short Clients: {args.short_clients}")
    print(f"Long Clients: {args.long_clients}")
    print(f"Sleep Time: {args.sleep} seconds")
    print(f"Local Port: {args.local_port}")
    print(f"Remote Port: {args.remote_port}")
    print("------------------------\n")

    servers = setup_vllm_servers(args.vllm_url, args.local_port, args.remote_port)

    # 创建结果队列
    result_queue = asyncio.Queue()

    # 创建文件写入协程
    async def write_results():
        results = []
        # 首次写入时清空文件
        with open('benchmark_results.json', 'w') as f:
            json.dump([], f)

        while True:
            result = await result_queue.get()
            if result is None:
                break
            results.append(result)
            # 实时写入文件
            with open('benchmark_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            result_queue.task_done()

    # 启动写入任务
    writer_task = asyncio.create_task(write_results())

    # 新增：启动监控任务
    monitor_task = setup_benchmark_monitoring(result_queue)

    short_configurations = []
    long_configurations = []
    # 创建范围时使用整数并设置步长为1
    for qps_value in np.arange(args.range_lower, args.range_higher + 0.1, 1):  # 加0.1是为了包含上限
        if args.short_qps == 0 and args.long_qps == 0:
            print("[Error]: short_qps and long_qps cannot be 0 at the same time")
            exit(1)
        qps_value = round(qps_value, 1)  # 防止浮点数精度问题
        if args.short_qps == 0:
            if abs(qps_value % args.long_qps) < 1e-6:
                configuration = {"num_requests": args.num_requests, "qps": args.range_lower, "output_tokens": 200}
                short_configurations.append(configuration)
        else:
            if abs(qps_value % args.short_qps) < 1e-6:
                configuration = {"num_requests": args.num_requests, "qps": qps_value, "output_tokens": 200}
                short_configurations.append(configuration)

        if args.long_qps == 0:
            if abs(qps_value % args.short_qps) < 1e-6:
                configuration = {"num_requests": args.num_requests, "qps": args.range_lower, "output_tokens": 200}
                long_configurations.append(configuration)
        else:
            if abs(qps_value % args.long_qps) < 1e-6:
                configuration = {"num_requests": args.num_requests, "qps": qps_value, "output_tokens": 200}
                long_configurations.append(configuration)

    print(f"short_configurations: {short_configurations}")
    print(f"long_configurations: {long_configurations}")

    # 记录开始时间
    start_time = time.time()

    # 创建所有基准测试任务
    tasks = []
    for index in range(args.short_clients):
        task = asyncio.create_task(
            run_all_benchmarks(
                args.local_port, args.api_key, args.distribution,
                short_configurations, args.concurrency, 'short',
                index, args.sleep
            )
        )
        tasks.append(task)

    for index in range(args.long_clients):
        task = asyncio.create_task(
            run_all_benchmarks(
                args.local_port, args.api_key, args.distribution,
                long_configurations, args.concurrency, 'long',
                index, args.sleep
            )
        )
        tasks.append(task)

    # 处理完成的任务结果
    for task in asyncio.as_completed(tasks):
        result = await task
        await result_queue.put(result)

    # 停止监控（在所有任务完成后）
    _BENCHMARK_MONITOR.stop_monitoring()
    await monitor_task

    # 记录结束时间
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time: {total_time:.2f} seconds")

    # 发送结束信号给写入协程
    await result_queue.put(None)
    # 等待写入任务完成
    await writer_task

    print("Benchmark results saved to benchmark_results.json")

    for server in servers:
        stop_tunnel(server)

    plot_result(args.concurrency, args.num_requests, round(total_time, 2))


if __name__ == "__main__":
    asyncio.run(main())

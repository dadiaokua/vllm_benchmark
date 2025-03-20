import asyncio
import json
import os
import time
import argparse
import traceback
from datetime import datetime

import numpy as np

from BenchmarkMonitor.BenchmarkMonitor import monitor_results
from plot.plot import plot_result
from util.util import open_jsonl_file, QAJsonFormatter, setup_vllm_servers, stop_tunnel
from vllm_benchmark import run_benchmark
from openai import AsyncOpenAI
from transformers import AutoTokenizer


async def run_all_benchmarks(local_port, api_key, distribution, request_timeout, configurations, concurrency,
                             client_type, round_time, index, sleep, result_queue, update_event, use_time_data):
    all_results = []
    # 创建多个client实例
    clients = []
    if isinstance(local_port, list):
        for port in local_port:
            clients.append(AsyncOpenAI(base_url="http://localhost:" + str(port) + "/v1"))
    else:
        clients.append(AsyncOpenAI(base_url="http://localhost:" + str(local_port) + "/v1"))

    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))

    time_data, data_path, jsonl_files = open_jsonl_file(client_type, dataset2prompt)

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
        results = await run_benchmark(concurrency, request_timeout, config['output_tokens'],
                                      clients, distribution, config['qps'], client_index, formatted_json, i, tokenizer,
                                      time_data, use_time_data, round_time)
        await result_queue.put(results)
        all_results.append(results)
        update_event.set()
        if results["successful_requests"] * 100 / results['total_requests'] < 70:
            # 如果成功率小于70%，后续配置都使用当前配置的qps
            current_qps = config['qps'] - 5
            for config_round, remaining_config in enumerate(configurations[i + 1:]):
                remaining_config['qps'] = current_qps
                results = await run_benchmark(concurrency, request_timeout,
                                              remaining_config['output_tokens'], clients, distribution,
                                              remaining_config['qps'], client_index, formatted_json,
                                              config_round + i + 1, tokenizer, time_data, use_time_data, round_time)
                await result_queue.put(results)
                all_results.append(results)
                update_event.set()
                time.sleep(sleep)
            break
        time.sleep(sleep)  # Wait a bit between runs to let the system cool down

    return all_results


async def main():
    parser = argparse.ArgumentParser(description="Run vLLM benchmarks with various configurations")
    parser.add_argument("--vllm_url", type=str, nargs='+', required=True,
                        help="URLs of the vLLM servers (can provide multiple)",
                        default=["http://127.0.0.1"])
    parser.add_argument("--api_key", type=str, required=True, help="API key for vLLM server", default='test')
    parser.add_argument("--distribution", type=str, help="Distribution of request")
    parser.add_argument("--short_qps", type=int, help="Qps of short request", required=True, default=1)
    parser.add_argument("--long_qps", type=int, help="Qps of long request", required=True, default=1)
    parser.add_argument("--range_lower", type=int, help="Lower", default=1)
    parser.add_argument("--range_higher", type=int, help="Higher", default=400)
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
    parser.add_argument("--round_time", type=int, default=600, help="Timeout for every round (default: 600)",required=True)

    args = parser.parse_args()

    print("\nBenchmark Configuration:")
    print("------------------------")
    print(f"vLLM Server URL: {args.vllm_url}")
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
    print(f"Use Time Data: {args.use_time_data}")
    print(f"Request Timeout: {args.request_timeout} seconds")
    print(f"Round Time: {args.round_time} seconds")
    print("------------------------\n")

    servers = setup_vllm_servers(args.vllm_url, args.local_port, args.remote_port)

    # 创建一个列表来存储所有结果
    all_results = []

    short_configurations = []
    long_configurations = []
    # 创建范围时使用整数并设置步长为1
    for qps_value in np.arange(args.range_lower, args.range_higher + 1.1, 1):  # 加0.1是为了包含上限
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
                configuration = {"num_requests": args.num_requests, "qps": int(qps_value), "output_tokens": 200}
                short_configurations.append(configuration)

        if args.long_qps == 0:
            if abs(qps_value % args.short_qps) < 1e-6:
                configuration = {"num_requests": args.num_requests, "qps": args.range_lower, "output_tokens": 200}
                long_configurations.append(configuration)
        else:
            if abs(qps_value % args.long_qps) < 1e-6:
                configuration = {"num_requests": args.num_requests, "qps": int(qps_value), "output_tokens": 200}
                long_configurations.append(configuration)

    print(f"short_configurations: {short_configurations}")
    print(f"long_configurations: {long_configurations}")

    # 记录开始时间
    start_time = time.time()
    all_results = asyncio.Queue()  # 用队列存储结果
    update_event = asyncio.Event()  # 用事件控制更新触发
    # 创建所有基准测试任务
    tasks = []

    # 创建监控任务
    monitor_task = asyncio.create_task(monitor_results(all_results, update_event, len(short_configurations), args.short_clients +  args.long_clients))
    tasks.append(monitor_task)

    try:
        # 设置任务超时时间为120分钟
        benchmark_timeout = 3600 * 2

        # 创建短请求任务
        for index in range(args.short_clients):
            task = asyncio.create_task(
                run_all_benchmarks(
                    args.local_port, args.api_key, args.distribution, args.request_timeout,
                    short_configurations, args.concurrency, 'short', args.round_time,
                    index, args.sleep, all_results, update_event, args.use_time_data
                )
            )
            tasks.append(task)

        # 创建长请求任务  
        for index in range(args.long_clients):
            task = asyncio.create_task(
                run_all_benchmarks(
                    args.local_port, args.api_key, args.distribution, args.request_timeout,
                    long_configurations, args.concurrency, 'long', args.round_time,
                    index, args.sleep, all_results, update_event, args.use_time_data
                )
            )
            tasks.append(task)

        # 等待所有任务完成,添加超时控制
        await asyncio.wait_for(asyncio.gather(*tasks[1:]), timeout=benchmark_timeout)

    except asyncio.TimeoutError:
        print(f"Tasks did not complete within {benchmark_timeout} seconds, cancelling...")
        # 取消所有未完成的任务
        for task in tasks:
            if not task.done():
                task.cancel()
        # 等待取消操作完成
        await asyncio.gather(*tasks, return_exceptions=True)
    except Exception as e:
        print(f"An error occurred: {e}")
        # 取消所有任务
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    # 从task的结果中获取所有结果
    all_benchmark_results = []
    for task in tasks[1:]:  # 跳过monitor_task
        if task.done():
            result = task.result()
            if result:  # 确保结果不为None
                all_benchmark_results.append(result)

    # 更新all_results变量
    benchmark_results = all_benchmark_results

    # 记录结束时间
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time: {total_time:.2f} seconds")

    start_datetime = datetime.fromtimestamp(start_time)
    end_datetime = datetime.fromtimestamp(end_time)

    # 生成文件名，拼接关键参数
    filename = (
        f"benchmark_start_{start_datetime.strftime("%H:%M")}_end_{end_datetime.strftime("%H:%M")}.json"
    )

    # 替换非法字符（确保文件名合法）
    filename = filename.replace(" ", "_").replace(":", "-").replace("/", "-")

    # 确保 results 目录存在
    os.makedirs("results", exist_ok=True)
    # 打印最终的文件名
    print(f"Saving benchmark results to: results/{filename}")

    # 所有任务完成后，一次性写入结果到文件
    with open("results/" + filename, 'w') as f:
        json.dump(benchmark_results, f, indent=2)

    for server in servers:
        stop_tunnel(server)

    plot_data = {
        "filename": filename,
        "concurrency": args.concurrency,
        "num_requests": args.num_requests,
        "total_time": round(total_time, 2)
    }

    with open("tmp_result/plot_data.json", "w") as f:
        json.dump(plot_data, f, indent=4)

    # 取消 monitor 任务
    monitor_task.cancel()

    # 等待 monitor_task 退出
    try:
        await monitor_task
    except asyncio.CancelledError:
        print("Monitor task cancelled.")

    plot_result(filename, args.concurrency, args.num_requests, round(total_time, 2))


if __name__ == "__main__":
    asyncio.run(main())

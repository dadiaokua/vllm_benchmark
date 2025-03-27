import asyncio
import json
import os
import time
import argparse
import traceback
from datetime import datetime

import numpy as np

from BenchmarkMonitor.BenchmarkMonitor import monitor_results, RESULTS_FILE
from plot.plot import plot_result
from util.util import open_jsonl_file, QAJsonFormatter, setup_vllm_servers, stop_tunnel
from vllm_benchmark import run_benchmark
from openai import AsyncOpenAI
from transformers import AutoTokenizer


async def run_all_benchmarks(local_port, api_key, distribution, request_timeout, configurations, concurrency,
                             client_type, round_time, index, sleep, result_queue, update_event, use_time_data):
    """Run benchmarks with given configurations and parameters"""
    # Initialize clients
    clients = initialize_clients(local_port)
    
    # Prepare data for benchmarking
    formatted_json = await prepare_benchmark_data(client_type)
    if not formatted_json:
        return None
        
    # Run benchmarks for each configuration
    client_index = f"{client_type}_{index}"
    all_results = await execute_benchmark_configurations(
        configurations, concurrency, request_timeout, clients, distribution,
        client_index, formatted_json, sleep, result_queue, update_event,
        client_type, round_time, use_time_data
    )
    
    return all_results


def initialize_clients(local_port):
    """Initialize OpenAI clients based on port configuration"""
    if isinstance(local_port, list):
        return [AsyncOpenAI(base_url=f"http://localhost:{port}/v1") for port in local_port]
    else:
        return [AsyncOpenAI(base_url=f"http://localhost:{local_port}/v1")]


async def prepare_benchmark_data(client_type):
    """Prepare and format data for benchmarking"""
    # Load dataset configuration
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    
    # Get data files
    time_data, data_path, jsonl_files = open_jsonl_file(client_type, dataset2prompt)
    
    # Initialize tokenizer
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained(
        '/Users/myrick/modelHub/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659',
        trust_remote_code=True)
    
    # Format and filter data
    try:
        formatter = QAJsonFormatter()
        max_samples = 500000 if client_type == 'long' else 100000
        formatted_json = await formatter.format_qa_json(
            tokenizer, dataset2prompt, 5000, jsonl_files, data_path, max_samples, client_type)
        
        # Filter by length
        filtered_json = []
        str_count = 0
        for item in formatted_json:
            if len(str(item)) < 4000:
                str_count += 1
                filtered_json.append(item)
        print(f'request count: {str_count}')
        
        return formatted_json
    except Exception as e:
        print(f"Error: {str(e)}")
        print("详细错误信息:")
        print(traceback.format_exc())
        return None


async def execute_benchmark_configurations(configurations, concurrency, request_timeout, clients, distribution,
                                          client_index, formatted_json, sleep, result_queue, update_event,
                                          client_type, round_time, use_time_data):
    """Execute benchmarks for all configurations"""
    all_results = []
    
    # Get tokenizer for benchmark
    tokenizer = AutoTokenizer.from_pretrained(
        '/Users/myrick/modelHub/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659',
        trust_remote_code=True)
    
    # Get time data
    time_data, _, _ = open_jsonl_file(client_type, json.load(open("config/dataset2prompt.json", "r")))
    
    for i, config in enumerate(configurations):
        # Run benchmark with current configuration
        results = await run_benchmark(
            concurrency, request_timeout, config['output_tokens'],
            clients, distribution, config['qps'], client_index, 
            formatted_json, i, tokenizer, time_data, use_time_data, round_time
        )
        
        # Store and update results
        await result_queue.put(results)
        all_results.append(results)
        update_event.set()
        
        # Check success rate and adjust if needed
        if should_reduce_qps(results):
            await handle_reduced_qps(
                results, configurations[i+1:], i, concurrency, request_timeout, clients,
                distribution, client_index, formatted_json, tokenizer, time_data,
                use_time_data, round_time, sleep, result_queue, update_event, all_results
            )
            break
            
        time.sleep(sleep)  # Wait between runs to let the system cool down

    return all_results


def should_reduce_qps(results):
    """Determine if QPS should be reduced based on success rate"""
    success_rate = results["successful_requests"] * 100 / results['total_requests']
    return success_rate < 70


async def handle_reduced_qps(current_results, remaining_configs, current_index, concurrency, request_timeout, 
                            clients, distribution, client_index, formatted_json, tokenizer, time_data,
                            use_time_data, round_time, sleep, result_queue, update_event, all_results):
    """Handle remaining configurations with reduced QPS"""
    current_qps = current_results['qps'] - 5
    
    for config_round, remaining_config in enumerate(remaining_configs):
        remaining_config['qps'] = current_qps
        results = await run_benchmark(
            concurrency, request_timeout, remaining_config['output_tokens'], 
            clients, distribution, current_qps, client_index, 
            formatted_json, config_round + current_index + 1, tokenizer, 
            time_data, use_time_data, round_time
        )
        await result_queue.put(results)
        all_results.append(results)
        update_event.set()
        time.sleep(sleep)


async def setup_benchmark_tasks(args, all_results, update_event, short_configurations, long_configurations):
    """Setup and create benchmark tasks"""
    tasks = []
    
    # Create monitor task
    monitor_task = asyncio.create_task(monitor_results(all_results, update_event, 
        len(short_configurations), args.short_clients + args.long_clients))
    tasks.append(monitor_task)

    # Create short request tasks
    for index in range(args.short_clients):
        task = asyncio.create_task(
            run_all_benchmarks(
                args.local_port, args.api_key, args.distribution, args.request_timeout,
                short_configurations, args.concurrency, 'short', args.round_time,
                index, args.sleep, all_results, update_event, args.use_time_data
            )
        )
        tasks.append(task)

    # Create long request tasks
    for index in range(args.long_clients):
        task = asyncio.create_task(
            run_all_benchmarks(
                args.local_port, args.api_key, args.distribution, args.request_timeout,
                long_configurations, args.concurrency, 'long', args.round_time,
                index, args.sleep, all_results, update_event, args.use_time_data
            )
        )
        tasks.append(task)

    return tasks, monitor_task

def generate_configurations(args):
    """Generate benchmark configurations"""
    short_configurations = []
    long_configurations = []
    
    for qps_value in np.arange(args.range_lower, args.range_higher + 1.1, 1):
        if args.short_qps == 0 and args.long_qps == 0:
            print("[Error]: short_qps and long_qps cannot be 0 at the same time")
            exit(1)
            
        qps_value = round(qps_value, 1)
        
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
                
    return short_configurations, long_configurations

def save_benchmark_results(filename, benchmark_results, plot_data):
    """Save benchmark results to files"""
    os.makedirs("results", exist_ok=True)
    print(f"Saving benchmark results to: results/{filename}")

    with open("results/" + filename, 'w') as f:
        json.dump(benchmark_results, f, indent=2)

    with open("tmp_result/plot_data.json", "w") as f:
        json.dump(plot_data, f, indent=4)

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

    with open(RESULTS_FILE, "w") as f:
        json.dump([], f)

    all_results = asyncio.Queue()
    update_event = asyncio.Event()
    
    short_configurations, long_configurations = generate_configurations(args)
    print(f"short_configurations: {short_configurations}")
    print(f"long_configurations: {long_configurations}")

    start_time = time.time()
    tasks, monitor_task = await setup_benchmark_tasks(args, all_results, update_event, 
                                                    short_configurations, long_configurations)

    try:
        benchmark_timeout = 3600 * 2
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

    benchmark_results = all_benchmark_results

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time: {total_time:.2f} seconds")

    start_datetime = datetime.fromtimestamp(start_time)
    end_datetime = datetime.fromtimestamp(end_time)

    filename = (
        f"benchmark_start_{start_datetime.strftime("%H:%M")}_end_{end_datetime.strftime("%H:%M")}.json"
    )
    filename = filename.replace(" ", "_").replace(":", "-").replace("/", "-")

    plot_data = {
        "filename": filename,
        "concurrency": args.concurrency,
        "num_requests": args.num_requests,
        "total_time": round(total_time, 2)
    }

    save_benchmark_results(filename, benchmark_results, plot_data)

    for server in servers:
        stop_tunnel(server)

    monitor_task.cancel()
    try:
        await monitor_task
    except asyncio.CancelledError:
        print("Monitor task cancelled.")

    plot_result(filename, args.concurrency, args.num_requests, round(total_time, 2))


if __name__ == "__main__":
    asyncio.run(main())

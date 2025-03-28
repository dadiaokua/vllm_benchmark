import json
import time
import numpy as np
from openai import AsyncOpenAI
from transformers import AutoTokenizer
from util.JsonFormatterUtil import open_jsonl_file
from vllm_benchmark import run_benchmark

async def run_all_benchmarks(local_port, api_key, distribution, request_timeout, configurations, concurrency,
                             formatted_json,
                             client_type, round_time, index, sleep, result_queue, update_event, use_time_data):
    """Run benchmarks with given configurations and parameters"""
    # Initialize clients
    clients = initialize_clients(local_port)

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

        # # Check success rate and adjust if needed
        # if should_reduce_qps(results):
        #     await handle_reduced_qps(
        #         results, configurations[i+1:], i, concurrency, request_timeout, clients,
        #         distribution, client_index, formatted_json, tokenizer, time_data,
        #         use_time_data, round_time, sleep, result_queue, update_event, all_results
        #     )
        #     break

        time.sleep(sleep)  # Wait between runs to let the system cool down

    return all_results

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
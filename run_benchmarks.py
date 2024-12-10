import asyncio
import json
import time
import argparse
from vllm_benchmark import run_benchmark
from openai import AsyncOpenAI

async def run_all_benchmarks(vllm_url, api_key, distribution, qps):
    configurations = []
    for i in range(200,501):
        if i % 50 == 0:
            configuration = {"num_requests": 1000, "concurrency": i, "output_tokens": 200}
            configurations.append(configuration)
        else:
            continue
    print(configurations)
    all_results = []

    client = AsyncOpenAI(base_url=vllm_url+"/v1")

    for config in configurations:
        print(f"Running benchmark with concurrency {config['concurrency']}...")
        results = await run_benchmark(config['num_requests'], config['concurrency'], 30, config['output_tokens'],
                                      client, api_key, distribution, qps)
        all_results.append(results)
        time.sleep(5)  # Wait a bit between runs to let the system cool down

    return all_results

def main():
    parser = argparse.ArgumentParser(description="Run vLLM benchmarks with various configurations")
    parser.add_argument("--vllm_url", type=str, required=True, help="URL of the vLLM server",
                        default="http://127.0.0.1")
    parser.add_argument("--api_key", type=str, required=True, help="API key for vLLM server", default='test')
    parser.add_argument("--use_long_context", action="store_true",
                        help="Use long context prompt pairs instead of short prompts", default=True)
    parser.add_argument("--distribution", type=str, help="Distribution of request")
    parser.add_argument("--qps", type=int, help="Qps of request", required=True, default=1)

    args = parser.parse_args()

    all_results = asyncio.run(run_all_benchmarks(args.vllm_url, args.api_key, args.distribution, args.qps))

    with open('benchmark_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print("Benchmark results saved to benchmark_results.json")

if __name__ == "__main__":
    main()


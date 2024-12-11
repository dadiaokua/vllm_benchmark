import asyncio
import json
import time
import argparse
from vllm_benchmark import run_benchmark
from openai import AsyncOpenAI


async def run_all_benchmarks(vllm_url, api_key, distribution, qps, range_lower, range_higher, concurrency,
                             num_requests):
    configurations = []
    if range_lower > concurrency | range_higher < concurrency:
        print(f"Concurrency must be between {range_lower} and {range_higher}")
        return None
    for i in range(range_lower, range_higher):
        if i % concurrency == 0:
            configuration = {"num_requests": num_requests, "concurrency": i, "output_tokens": 200}
            configurations.append(configuration)
        else:
            continue
    print(configurations)
    all_results = []

    client = AsyncOpenAI(base_url=vllm_url + "/v1")

    for config in configurations:
        print(f"Running benchmark with concurrency {config['concurrency']}...")
        results = await run_benchmark(config['num_requests'], config['concurrency'], 30, config['output_tokens'],
                                      client, api_key, distribution, qps)
        all_results.append(results)
        time.sleep(5)  # Wait a bit between runs to let the system cool down

    return all_results


async def main():
    parser = argparse.ArgumentParser(description="Run vLLM benchmarks with various configurations")
    parser.add_argument("--vllm_url", type=str, required=True, help="URL of the vLLM server",
                        default="http://127.0.0.1")
    parser.add_argument("--api_key", type=str, required=True, help="API key for vLLM server", default='test')
    parser.add_argument("--use_long_context", action="store_true",
                        help="Use long context prompt pairs instead of short prompts", default=True)
    parser.add_argument("--distribution", type=str, help="Distribution of request")
    parser.add_argument("--qps", type=int, help="Qps of request", required=True, default=1)
    parser.add_argument("--range_lower", type=int, help="Lower", default=1)
    parser.add_argument("--range_higher", type=int, help="Higher", default=1001)
    parser.add_argument("--concurrency", type=int, help="concurrency", default=10)
    parser.add_argument("--num_requests", type=int, help="Number of requests", default=1000)
    parser.add_argument("--clients", type=int, help="Number of requests", default=1)

    args = parser.parse_args()

    # 创建多个异步任务
    tasks = [
        run_all_benchmarks(
            args.vllm_url, args.api_key, args.distribution,
            args.qps, args.range_lower, args.range_higher,
            args.concurrency, args.num_requests
        )
        for _ in range(args.clients)
    ]

    # 并发执行所有任务
    all_results = await asyncio.gather(*tasks)

    # all_results = []
    #
    # for i in range(args.clients):
    #     all_result = asyncio.run(run_all_benchmarks(args.vllm_url, args.api_key, args.distribution,
    #         args.qps, args.range_lower, args.range_higher, args.concurrency, args.num_requests))
    #     if all_result == None:
    #         return
    #     else:
    #         all_results.append(all_result)

    with open('benchmark_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print("Benchmark results saved to benchmark_results.json")


if __name__ == "__main__":
    asyncio.run(main())

import asyncio
import random
import time

from config.Config import GLOBAL_CONFIG
from util.MathUtil import calculate_metrics
from util.RequestUtil import worker


class BenchmarkClient:
    """Class representing a benchmark client with its configurations and state"""

    def __init__(self, client_type, client_index, qps, port, api_key, tokenizer,
                 distribution, request_timeout, concurrency, round, round_time, sleep, time_data,
                 result_queue, update_event, formatted_json, OpenAI_client, use_time_data=0):
        """Initialize a benchmark client

        Args:
            client_type (str): Type of client ('short' or 'long')
            client_index (int): Index of this client
            configurations (list): List of benchmark configurations
            port (list): List of local ports
            api_key (str): API key for vLLM server
            distribution (str): Distribution of requests
            request_timeout (int): Timeout for each request in seconds
            concurrency (int): Number of concurrent requests
            round_time (int): Timeout for every round in seconds
            sleep (int): Sleep time between rounds
            result_queue (asyncio.Queue): Queue for sending results
            update_event (asyncio.Event): Event for notifying monitor
            use_time_data (int): Whether to use time data
            formatted_json (list): Formatted input JSON data
        """
        self.client_type = client_type
        self.client_index = client_index
        self.client_id = f"{client_type}_{client_index}"
        self.qps = qps
        self.port = port
        self.api_key = api_key
        self.distribution = distribution
        self.request_timeout = request_timeout
        self.concurrency = concurrency
        self.round_time = round_time
        self.sleep = sleep
        self.result_queue = result_queue
        self.update_event = update_event
        self.use_time_data = use_time_data
        self.formatted_json = formatted_json
        self.tokenizer = tokenizer
        self.time_data = time_data
        self.round = round
        self.latency_slo = random.randint(8, 15)

        self.avg_latency_div_standard_latency = -1
        self.slo_violation_count = -1
        self.service = -1
        self.service_div_latency = -1
        self.exchange_Resources_Times = 0
        self.active_ratio = 1.0
        self.fairness_ratio = -1
        self.credit = 0

        self.openAI_client = OpenAI_client
        self.monitor_done_event = asyncio.Event()

        # State tracking
        self.results = []
        self.task = None

    async def run_all_benchmarks(self):
        """Run all benchmark configurations for this client"""
        print(f"Starting benchmarks for client {self.client_id} with {self.round} configurations")

        for i in range(self.round):
            # Run benchmark with current configuration
            print(f"Client {self.client_id}: Running configuration {i + 1}/{self.round}: {self.qps}")
            result = await self.run_benchmark(GLOBAL_CONFIG["output_tokens"], self.qps, i, self.latency_slo)

            # Store and update results
            self.results.append(result)
            await self.result_queue.put(1)
            self.update_event.set()

            # 等待 monitor 通知处理完成
            await self.monitor_done_event.wait()
            self.monitor_done_event.clear()

            # Give monitor time to process
            await asyncio.sleep(1)

            # Wait between runs
            await asyncio.sleep(self.sleep)

        return self.results

    async def run_benchmark(self, output_tokens, qps, config_round, latency_slo):
        print(f"[Client {self.client_id}] Starting benchmark run with QPS={qps}, output_tokens={output_tokens}")
        assert self.openAI_client is not None, "OpenAI Client must not be None"

        semaphore = asyncio.Semaphore(self.concurrency)
        results = []
        workers = []
        start_time = time.time()
        print(f"[Client {self.client_id}] Benchmark started at {start_time}")

        qps_per_worker = round(qps / self.concurrency)
        if qps < self.concurrency:
            print(
                f"[Client {self.client_id}] QPS ({qps}) < concurrency ({self.concurrency}), adjusting concurrency to 1")
            qps_per_worker = qps
            self.concurrency = 1

        print(f"[Client {self.client_id}] Creating {self.concurrency} workers with {qps_per_worker} QPS per worker")
        for worker_id in range(self.concurrency):
            worker_task = asyncio.create_task(
                worker(self.openAI_client, semaphore, results, output_tokens, self.client_id, self.tokenizer,
                       self.request_timeout, self.round_time, qps_per_worker, self.distribution, self.formatted_json,
                       config_round, worker_id, self.time_data, self.use_time_data, latency_slo, self.active_ratio, )
            )
            workers.append(worker_task)

        print(f"[Client {self.client_id}] Waiting for all workers to complete")
        worker_results = await asyncio.gather(*workers)

        num_requests = sum(worker_results)
        print(f"[Client {self.client_id}] Total requests completed: {num_requests}")

        end_time = time.time()
        print(f"[Client {self.client_id}] Benchmark ended at {end_time}, total time: {end_time - start_time:.2f}s")

        result = calculate_metrics(self.concurrency, self.request_timeout, self.client_id, results, start_time,
                                   end_time, num_requests, qps, output_tokens, latency_slo)
        self.avg_latency_div_standard_latency = result['avg_latency_div_standard_latency']
        self.slo_violation_count = result['slo_violation_count']

        print(
            f"[Client {self.client_id}] Benchmark complete. Avg latency ratio: {self.avg_latency_div_standard_latency:.2f}, SLO violations: {self.slo_violation_count}")
        return result

    def start(self):
        """Start the benchmark task"""
        self.task = asyncio.create_task(self.run_all_benchmarks())
        return self.task

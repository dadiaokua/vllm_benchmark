import asyncio
import random

from util.RequestUtil import run_benchmark


class BenchmarkClient:
    """Class representing a benchmark client with its configurations and state"""

    def __init__(self, client_type, client_index, configurations, port, api_key, tokenizer,
                 distribution, request_timeout, concurrency, round_time, sleep, time_data,
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
        self.configurations = configurations
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
        self.latency_slo = random.randint(0, 10)

        self.openAI_client = OpenAI_client

        # State tracking
        self.current_config_index = 0
        self.results = []
        self.task = None

    def get_current_config(self):
        """Get the current configuration"""
        if self.current_config_index < len(self.configurations):
            return self.configurations[self.current_config_index]
        return None

    def advance_config(self):
        """Move to the next configuration"""
        self.current_config_index += 1
        return self.get_current_config()

    def reduce_qps(self, reduction_factor=0.8):
        """Reduce QPS for remaining configurations"""
        for i in range(self.current_config_index, len(self.configurations)):
            self.configurations[i]["qps"] = int(self.configurations[i]["qps"] * reduction_factor)
        print(f"Reduced QPS for client {self.client_id} by factor {reduction_factor}")
        print(f"New configurations: {self.configurations[self.current_config_index:]}")

    async def run_all_benchmarks(self):
        """Run all benchmark configurations for this client"""
        print(f"Starting benchmarks for client {self.client_id} with {len(self.configurations)} configurations")

        for i, config in enumerate(self.configurations):
            self.current_config_index = i

            # Run benchmark with current configuration
            print(f"Client {self.client_id}: Running configuration {i + 1}/{len(self.configurations)}: {config}")
            result = await run_benchmark(
                self, self.concurrency, self.request_timeout, config['output_tokens'], self.openAI_client,
                self.distribution, config['qps'], self.client_id,
                self.formatted_json, i, self.tokenizer, self.time_data, self.use_time_data, self.round_time
            )

            # Store and update results
            self.results.append(result)
            await self.result_queue.put(1)
            self.update_event.set()

            # Give monitor time to process
            await asyncio.sleep(1)

            # Wait between runs
            await asyncio.sleep(self.sleep)

        return self.results

    def start(self):
        """Start the benchmark task"""
        self.task = asyncio.create_task(self.run_all_benchmarks())
        return self.task

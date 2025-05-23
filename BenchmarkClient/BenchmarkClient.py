import asyncio
import logging
import os
import random

from config.Config import GLOBAL_CONFIG
from experiment.base_experiment import BaseExperiment

from experiment.FCFS_experiment import FCFSExperiment
from experiment.LFS_experiment import LFSExperiment
from experiment.VTC_experiment import VTCExperiment


class BenchmarkClient:
    """Class representing a benchmark client with its configurations and state"""

    def __init__(self, client_type, client_index, qps, port, api_key, tokenizer, exp_type,
                 distribution, request_timeout, concurrency, round, round_time, sleep, time_data,
                 result_queue, formatted_json, OpenAI_client, qps_ratio, latency_slo, use_time_data=0):
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
        self.qps_ratio = qps_ratio
        self.port = port
        self.api_key = api_key
        self.distribution = distribution
        self.request_timeout = request_timeout
        self.concurrency = concurrency
        self.round_time = round_time
        self.sleep = sleep
        self.result_queue = result_queue
        self.use_time_data = use_time_data
        self.formatted_json = formatted_json
        self.tokenizer = tokenizer
        self.time_data = time_data
        self.round = round
        self.exp_type = exp_type
        self.latency_slo = latency_slo

        self.avg_latency_div_standard_latency = -1
        self.slo_violation_count = -1
        self.service = -1
        self.service_div_latency = -1
        self.exchange_Resources_Times = 0
        self.active_ratio = 1.0
        self.time_ratio = 1.0
        self.fairness_ratio = 0
        self.credit = 0

        self.openAI_client = OpenAI_client
        self.monitor_done_event = asyncio.Event()

        # State tracking
        self.results = []
        self.task = None

        self.experiment_config = None
        self.experiment = None

        # 设置logger（只设置一次，防止重复handler）
        self.logger = self._setup_logger()

    def _setup_logger(self):
        # 日志文件夹和文件名
        log_dir = "log"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"client_{self.client_id}_run.log")

        logger = logging.getLogger(f"client_{self.client_id}")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            fh = logging.FileHandler(log_file, encoding="utf-8")
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

    async def run_all_benchmarks(self):
        """Run all benchmark configurations for this client"""
        print(f"Starting benchmarks for client {self.client_id} with {self.round} configurations")

        for i in range(self.round):
            # Run benchmark with current configuration
            self.qps = self.qps * self.qps_ratio
            print(f"Client {self.client_id}: Running configuration {i + 1}/{self.round}: {self.qps}")
            result = await self.run_benchmark(GLOBAL_CONFIG["output_tokens"], self.qps, i, self.latency_slo)

            if i != 0:
                # 等待 monitor 通知处理完成
                await self.monitor_done_event.wait()
                self.monitor_done_event.clear()
                if i == 1:
                    self.results[-1]["fairness_ratio"] = self.fairness_ratio


            # Store and update results
            if result:
                self.results.append(result)
            else:
                self.results.append(self.results[-1])
            await self.result_queue.put(1)

            self.results[-1]["fairness_ratio"] = self.fairness_ratio
            # Give monitor time to process
            await asyncio.sleep(1)

            # Wait between runs
            await asyncio.sleep(self.sleep)

        return self.results

    async def run_benchmark(self, output_tokens, qps, config_round, latency_slo):
        """
        运行基准测试实验

        Args:
            output_tokens: 每个请求的输出令牌数
            qps: 每秒查询数
            config_round: 配置轮次
            latency_slo: 延迟服务水平目标

        Returns:
            dict: 实验结果指标
        """

        self.experiment_config = {
            'output_tokens': output_tokens,
            'qps': qps,
            'config_round': config_round,
            'latency_slo': latency_slo
        }

        experiment_types = {
            "baseline": BaseExperiment,
            "LFS": LFSExperiment,
            "VTC": VTCExperiment,
            "FCFS": FCFSExperiment
        }

        # 创建并运行实验
        experiment_class = experiment_types.get(self.exp_type, BaseExperiment)
        self.experiment = experiment_class(self)
        await self.experiment.setup()
        result = await self.experiment.run()

        return result

    def start(self):
        """Start the benchmark task"""
        self.task = asyncio.create_task(self.run_all_benchmarks())
        return self.task

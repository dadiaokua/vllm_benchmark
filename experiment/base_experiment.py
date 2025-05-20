import asyncio
import random
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

from util.JsonFormatterUtil import make_prefix_list
from util.MathUtil import calculate_metrics
from util.RequestUtil import worker


class BaseExperiment:
    """
    表示单个基准测试实验的类
    负责设置、运行和收集实验结果
    """

    def __init__(self, client, config):
        """
        初始化基准测试实验

        Args:
            client: 运行实验的客户端实例
            config: 实验配置参数
        """
        self.client = client
        self.config = config

        # 从客户端获取必要的属性
        self.client_id = client.client_id
        self.openAI_client = client.openAI_client
        self.concurrency = client.concurrency
        self.tokenizer = client.tokenizer
        self.request_timeout = client.request_timeout
        self.round_time = client.round_time
        self.distribution = client.distribution
        self.formatted_json = client.formatted_json
        self.time_data = client.time_data
        self.use_time_data = client.use_time_data

        # 实验特定参数
        self.output_tokens = config.get('output_tokens', 200)
        self.qps = config.get('qps', 1)
        self.config_round = config.get('config_round', 1)
        self.latency_slo = config.get('latency_slo', 10)
        self.active_ratio = client.active_ratio
        self.time_ratio = client.time_ratio

        self.qps_per_worker = 0
        self.exp_type = ''
        self.drift_time = 0

        # 实验结果
        self.results = []
        self.start_time = None
        self.end_time = None
        self.num_requests = 0
        self.metrics = None

    async def setup(self):
        """设置实验，进行必要的准备工作"""
        assert self.openAI_client is not None, "OpenAI Client must not be None"

        print(
            f"[Client {self.client_id}] Experiment setup complete: {self.qps} workers with {1} QPS per worker")
        return self

    async def run(self):
        """运行实验并收集结果"""
        print(
            f"[Client {self.client_id}] Starting benchmark run with QPS={self.qps}, output_tokens={self.output_tokens}")

        # 创建信号量控制并发
        semaphore = asyncio.Semaphore(self.concurrency)
        self.results = []
        workers = []

        # 记录开始时间
        self.start_time = time.time()
        print(
            f"[Client {self.client_id}] Benchmark started at {datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S')}")

        # 根据并发数创建workers
        if self.qps < self.concurrency:
            worker_counts = 1
        else:
            worker_counts = self.concurrency
        qps_per_worker = self.qps / worker_counts
        print(f"[Client {self.client_id}] Creating {worker_counts} workers, each handling {qps_per_worker} QPS")

        # Split formatted_json into equal chunks based on concurrency
        chunk_size = len(self.formatted_json) // worker_counts
        remaining = len(self.formatted_json) % worker_counts
        total_size = len(self.formatted_json)

        for worker_id in range(worker_counts):
            # Randomly select a starting point
            start_idx = random.randint(0, total_size - chunk_size - (1 if remaining > 0 else 0))
            end_idx = start_idx + chunk_size + (1 if remaining > 0 else 0)
            remaining = max(0, remaining - 1)  # Decrement remaining if needed

            # Handle wrap-around case if end_idx exceeds list length
            if end_idx > total_size:
                worker_json = self.formatted_json[start_idx:] + self.formatted_json[:end_idx - total_size]
            else:
                worker_json = self.formatted_json[start_idx:end_idx]
            if self.exp_type == "DLPM":
                worker_json = make_prefix_list(worker_json, self.tokenizer, 200 if "short" in self.client_id else 1000)
            else:
                random.shuffle(worker_json)

            worker_task = asyncio.create_task(
                worker(
                    self.openAI_client,
                    semaphore,
                    self.results,
                    self.output_tokens,
                    self.client_id,
                    self.tokenizer,
                    self.request_timeout,
                    self.round_time,
                    qps_per_worker,  # 每个worker处理相同比例的QPS
                    self.distribution,
                    worker_json,
                    self.config_round,
                    worker_id,
                    self.time_data,
                    self.use_time_data,
                    self.latency_slo,
                    self.time_ratio
                )
            )
            workers.append(worker_task)

        # 等待所有工作线程完成
        print(f"[Client {self.client_id}] Waiting for all workers to complete")
        worker_results = await asyncio.gather(*workers)
        completed_requests, drift_time, total_requests = zip(*worker_results)

        # 汇总结果
        self.num_requests = sum(completed_requests)
        self.drift_time = sum(drift_time)
        print(f"[Client {self.client_id}] Total requests completed: {self.num_requests}")

        # 记录结束时间
        self.end_time = time.time()
        print(
            f"[Client {self.client_id}] Benchmark ended at {datetime.fromtimestamp(self.end_time).strftime('%Y-%m-%d %H:%M:%S')}, total time: {self.end_time - self.start_time:.2f}s")

        # 计算指标
        return await self.calculate_results(completed_requests / total_requests)

    async def calculate_results(self, completed_requests_rate):
        """计算实验结果指标"""
        if not self.results or self.start_time is None or self.end_time is None:
            raise ValueError("Cannot calculate results: experiment has not been run")

        self.metrics = calculate_metrics(
            self.concurrency,
            self.request_timeout,
            self.client_id,
            self.results,
            self.start_time,
            self.end_time,
            self.num_requests,
            self.qps,
            self.output_tokens,
            self.latency_slo,
            self.client.fairness_ratio,
            self.drift_time,
            self.client.credit
        )

        # 更新客户端的相关指标
        self.client.avg_latency_div_standard_latency = self.metrics['avg_latency_div_standard_latency']
        self.client.slo_violation_count = self.metrics['slo_violation_count']

        # 如果成功率小于80%，则将QPS增加率设置为1
        if completed_requests_rate < 0.8:
            self.client.qps_ratio = 1

        return self.metrics

    def get_summary(self):
        """获取实验结果摘要"""
        if not self.metrics:
            return {"status": "not_run", "client_id": self.client_id}

        return {
            "client_id": self.client_id,
            "qps": self.qps,
            "output_tokens": self.output_tokens,
            "latency_slo": self.latency_slo,
            "total_requests": self.num_requests,
            "duration": self.end_time - self.start_time if self.end_time and self.start_time else 0,
            "avg_latency_ratio": self.metrics['avg_latency_div_standard_latency'],
            "slo_violations": self.metrics['slo_violation_count'],
            "success_rate": self.metrics['successful_requests'] / self.metrics['total_requests'] if self.metrics[
                                                                                                        'total_requests'] > 0 else 0
        }

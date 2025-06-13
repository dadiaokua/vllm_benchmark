import asyncio
import time
from datetime import datetime

from experiment.base_experiment import BaseExperiment
from util.RequestUtil import worker_with_queue
from RequestQueueManager.RequestQueueManager import RequestQueueManager, QueueStrategy


class QueueExperiment(BaseExperiment):
    """
    使用队列管理器的实验类
    """

    def __init__(self, client, queue_manager: RequestQueueManager = None, queue_strategy: QueueStrategy = QueueStrategy.FIFO):
        """
        初始化队列实验

        Args:
            client: 运行实验的客户端实例
            queue_manager: 可选的队列管理器实例，如果为None会自动创建
            queue_strategy: 队列调度策略
        """
        super().__init__(client)
        
        # 队列管理器
        self.queue_manager = queue_manager
        self.queue_strategy = queue_strategy
        self.queue_workers = 3  # 队列处理worker数量
        
        # 如果没有提供队列管理器，创建一个新的
        if self.queue_manager is None:
            self.queue_manager = RequestQueueManager(strategy=queue_strategy)
            self.queue_manager.set_openai_client(self.openAI_client)
        
        self.logger.info(f"QueueExperiment initialized with strategy: {queue_strategy.value}")

    async def setup(self):
        """设置实验，进行必要的准备工作"""
        await super().setup()
        
        # 确保队列管理器有OpenAI客户端
        if not self.queue_manager.openai_client:
            self.queue_manager.set_openai_client(self.openAI_client)
        
        # 启动队列管理器（如果还没有启动）
        if not self.queue_manager.is_running:
            # 在后台启动队列处理
            asyncio.create_task(self.queue_manager.start_processing(self.queue_workers))
            
            # 等待一下确保队列管理器启动
            await asyncio.sleep(0.5)
        
        self.logger.info(f"Queue experiment setup complete with {self.queue_workers} queue workers")
        return self

    async def run(self, config_round):
        """运行实验并收集结果"""

        self.logger.info(f"Starting queue-based benchmark round {config_round} run with QPS={self.qpm}, output_tokens={self.output_tokens}")
        self.logger.info(f"Client ID: {self.client_id}, Concurrency: {self.concurrency}")
        self.logger.info(f"Queue strategy: {self.queue_strategy.value}")
        self.logger.info(f"Time ratio: {self.time_ratio}, Active ratio: {self.active_ratio}, QPS ratio: {self.client.qpm_ratio}")

        # 创建信号量控制并发
        semaphore = asyncio.Semaphore(self.concurrency)
        self.experiment_results = []
        workers = []

        # 记录开始时间
        self.start_time = time.time()
        self.logger.info(f"Benchmark started at {datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S')}")

        # 根据并发数创建workers
        if self.qpm < self.concurrency:
            worker_counts = 1
        else:
            worker_counts = self.concurrency
        qpm_per_worker = self.qpm / worker_counts
        self.logger.info(f"Creating {worker_counts} workers, each handling {qpm_per_worker} QPS")
        self.logger.info(f"Total formatted JSON size: {len(self.formatted_json)}")

        # Split formatted_json into equal chunks based on concurrency
        chunk_size = len(self.formatted_json) // worker_counts
        remaining = len(self.formatted_json) % worker_counts
        total_size = len(self.formatted_json)
        self.logger.info(f"Chunk size: {chunk_size}, Remaining: {remaining}")

        start_index = 0
        for worker_id in range(worker_counts):
            # Calculate chunk size for this worker
            current_chunk_size = chunk_size + (1 if worker_id < remaining else 0)
            end_index = start_index + current_chunk_size

            # Extract chunk for this worker
            worker_json = self.formatted_json[start_index:end_index]
            start_index = end_index

            self.logger.info(f"Worker {worker_id}: processing {len(worker_json)} requests (indices {start_index - current_chunk_size} to {start_index - 1})")

            # 使用队列管理器的worker
            worker_task = asyncio.create_task(
                worker_with_queue(
                    self,
                    self.queue_manager,
                    semaphore,
                    self.experiment_results,
                    worker_id,
                    worker_json,
                    qpm_per_worker
                )
            )
            workers.append(worker_task)

        self.logger.info("Waiting for all workers to complete")
        worker_results = await asyncio.gather(*workers)
        completed_requests, drift_time, total_requests = zip(*worker_results)

        # 汇总结果
        self.num_requests = sum(completed_requests)
        self.drift_time = sum(drift_time)
        self.logger.info(f"Total requests succeeded: {self.num_requests}")

        # 记录结束时间
        self.end_time = time.time()
        self.logger.info(
            f"Benchmark ended at {datetime.fromtimestamp(self.end_time).strftime('%Y-%m-%d %H:%M:%S')}, total time: {self.end_time - self.start_time:.2f}s")

        # 打印队列统计信息
        self.queue_manager.print_statistics()

        # 计算指标
        return await self.calculate_results(sum(completed_requests) / sum(total_requests))

    async def cleanup(self):
        """清理资源"""
        if self.queue_manager and self.queue_manager.is_running:
            await self.queue_manager.cleanup()
            self.logger.info("Queue manager cleaned up")

    async def end(self):
        if self.queue_manager and self.queue_manager.is_running:
            await self.queue_manager.stop()
            self.logger.info("Queue manager stopped")

    def get_queue_statistics(self):
        """获取队列统计信息"""
        return self.queue_manager.get_statistics() if self.queue_manager else {} 
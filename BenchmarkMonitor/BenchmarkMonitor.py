import asyncio
import json
import subprocess
from datetime import datetime, timedelta

from config.Config import GLOBAL_CONFIG
from util.FileSaveUtil import save_results
from util.MathUtil import fairness_result, is_fairness_LFSLLM, is_fairness_VTC, is_fairness_DLPM

RESULTS_FILE = 'tmp_result/tmp_fairness_result.json'


class ExperimentMonitor:
    """
    实验监控器类，负责监控实验结果、计算公平性并触发资源调整
    """

    def __init__(self, clients, result_queue, client_count, exp_type, request_queue, config=None):
        """
        初始化监控器

        Args:
            clients: 客户端列表
            result_queue: 结果队列
            client_count: 客户端数量
            exp_type: 实验类型 (LFS, VTC, DLPM等)
            config: 配置参数，默认使用GLOBAL_CONFIG
        """
        self.clients = clients
        self.result_queue = result_queue
        self.request_queue = request_queue
        self.client_count = client_count
        self.exp_type = exp_type
        self.config = config or GLOBAL_CONFIG
        self.tmp_results = []
        self.fairness_results = []
        self.start_time = None
        self.logger = self._setup_logger()

        # 设置公平性调整策略映射
        self.fairness_strategies = {
            "LFS": is_fairness_LFSLLM,
            "VTC": is_fairness_VTC,
            "DLPM": is_fairness_DLPM
        }

    def _setup_logger(self):
        """设置日志记录器"""
        import logging
        logger = logging.getLogger(f"ExperimentMonitor-{self.exp_type}")
        logger.setLevel(logging.INFO)

        # 创建控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # 创建格式化器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)

        # 添加处理器到日志记录器
        logger.addHandler(ch)

        return logger

    async def __call__(self):
        """
        使类实例可以作为协程调用
        这样可以直接将实例传递给 asyncio.create_task()
        """
        return await self.start_monitoring()

    async def start_monitoring(self):
        """开始监控实验结果"""
        self.start_time = datetime.now()
        self.logger.info(f'Starting monitor for {self.exp_type} experiment with {self.client_count} clients')
        exp_duration = timedelta(seconds=self.config['exp_time'])

        while datetime.now() - self.start_time < exp_duration:
            await self._check_results()
            self._log_gpu_status()
            await asyncio.sleep(5)  # 每5秒检查一次

        self.logger.info(f'Experiment duration reached. Monitoring stopped.')
        return self.fairness_results

    def _log_gpu_status(self):
        """记录当前GPU利用率和显存占用"""
        try:
            result = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True
            )
            output = result.stdout.strip()
            log_lines = []
            for line in output.split('\n'):
                timestamp, index, util, mem_used, mem_total = [x.strip() for x in line.split(',')]
                log_line = (f"[{timestamp}] GPU {index} | Utilization: {util}% | "
                            f"Memory: {mem_used}/{mem_total} MiB")
                self.logger.info(log_line)
                log_lines.append(log_line)

            # 可选：追加记录到文件
            with open("tmp_result/gpu_monitor_log.txt", "a") as f:
                for log in log_lines:
                    f.write(log + "\n")

        except Exception as e:
            self.logger.warning(f"Failed to fetch GPU status: {e}")

    async def _check_request_queue(self):
        requests = []
        while not self.request_queue.empty():
            request = self.request_queue.get()
            requests.append(request)
        return requests

    async def _check_results(self):
        """检查结果队列并处理结果"""
        if not self.result_queue.empty():
            self.logger.info('Queue not empty, getting next result...')
            try:
                result = await asyncio.wait_for(self.result_queue.get(), timeout=10)
                self.tmp_results.append(result)
                self.logger.info(f'Current completed client numbers: {len(self.tmp_results)}')

                if len(self.tmp_results) == self.client_count:
                    await self._process_complete_round()

                self.result_queue.task_done()
                self.logger.info('Task marked as done')
            except asyncio.TimeoutError:
                self.logger.warning("Timeout while waiting for result.")
        else:
            self.logger.info(f'Queue is empty, waiting... (current client results: {len(self.tmp_results)})')

    async def _process_complete_round(self):
        """处理完整一轮的结果"""
        self.logger.info(f'Reached client_count={self.client_count}, calculating fairness...')

        # 计算公平性
        f_result, s_result = await self._calculate_fairness()
        self.logger.info(f"Fairness calculation complete. Fairness index: {f_result}")

        # 根据配置决定是否进行公平性调整
        if self.config["whether_fairness"]:
            exchange_count = await self._adjust_fairness()
        else:
            exchange_count = 0
            self.logger.info("Skipping fairness adjustment (disabled in config)")

        # 保存结果
        self._save_results(f_result, s_result, exchange_count)

        # 重置客户端
        await self._reset_clients()

        # 清空临时结果
        self.tmp_results = []

    async def _calculate_fairness(self):
        """计算公平性指标"""
        self.logger.info("Starting fairness calculation...")
        return await fairness_result(self.clients, self.exp_type)

    async def _adjust_fairness(self):
        """根据实验类型调整公平性"""
        self.logger.info(f"Starting fairness adjustment for {self.exp_type}...")
        
        # 从映射中获取调整函数
        adjust_function = self.fairness_strategies.get(self.exp_type)
        
        if adjust_function:
            exchange_count = await adjust_function(self.clients, self.exp_type)
        else:
            exchange_count = 0
            self.logger.warning(f"Invalid experiment type: {self.exp_type}, skipping fairness")
        
        self.logger.info("Fairness adjustment complete")
        return exchange_count

    def _save_results(self, f_result, s_result, exchange_count):
        """保存结果到文件"""
        results_file = self.config.get('RESULTS_FILE', RESULTS_FILE)
        save_results(exchange_count, f_result, s_result, results_file)
        self.fairness_results.append((f_result, s_result))
        self.logger.info(f'Results saved to {results_file}')

    async def _reset_clients(self):
        """重置所有客户端"""
        self.logger.info("Notifying clients of completion...")
        for i, client in enumerate(self.clients):
            self.logger.info(f"Resetting client {i + 1}/{len(self.clients)}")
            client.exchange_Resources_Times = 0
            client.monitor_done_event.set()
        self.logger.info("All clients notified")

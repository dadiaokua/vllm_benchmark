import asyncio
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
from util.RequestUtil import make_request


class QueueStrategy(Enum):
    """队列调度策略"""
    FIFO = "fifo"  # 先进先出
    PRIORITY = "priority"  # 基于优先级
    ROUND_ROBIN = "round_robin"  # 轮询
    SHORTEST_JOB_FIRST = "sjf"  # 最短作业优先
    FAIR_SHARE = "fair_share"  # 公平共享
    VTC = "vtc"


@dataclass
class  QueuedRequest:
    """队列中的请求对象"""
    start_time: float
    client_id: str
    worker_id: str
    request_content: str
    experiment: Any
    priority: int = 0
    submit_time: float = 0
    client_type: str = "unknown"  # short or long
    
    def __post_init__(self):
        if self.submit_time == 0:
            self.submit_time = time.time()
    
    def __lt__(self, other):
        """用于优先队列排序"""
        return self.priority < other.priority


class RequestQueueManager:
    """请求队列管理器，负责控制所有客户端请求的顺序"""
    
    def __init__(self, strategy: QueueStrategy = QueueStrategy.FIFO, max_queue_size: int = 10000):
        self.strategy = strategy
        self.max_queue_size = max_queue_size
        self.request_queue = asyncio.Queue(maxsize=max_queue_size)
        self.response_queues: Dict[str, asyncio.Queue] = {}  # 每个客户端的响应队列
        self.client_stats: Dict[str, Dict] = {}  # 客户端统计信息
        self.client_token_stats: Dict[str, Dict] = {}  # 每个客户端的token统计
        self.is_running = False
        self.workers_running = False
        self.openai_client = None
        self.logger = self._setup_logger()
        
        # 不同策略的特定数据结构
        self.priority_queue_list = []  # 改为列表，用于部分优先级
        self.round_robin_index = 0  # 轮询索引
        self.client_request_counts: Dict[str, int] = {}  # 每个客户端的请求计数
        
        # 部分优先级配置
        self.priority_insert_multiplier = 1  # 优先级倍数，优先级N可以往前插N*multiplier个位置
        self.max_priority_positions = 100  # 最大优先级插入位置限制
        
        # 统计信息
        self.total_requests_processed = 0
        self.start_time = None
        
    def _setup_logger(self):
        """设置日志记录器"""
        logger = logging.getLogger("RequestQueueManager")
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            logger.addHandler(ch)
            logger.propagate = False
        return logger
    
    def set_openai_client(self, client):
        """设置OpenAI客户端"""
        self.openai_client = client
    
    def configure_partial_priority(self, insert_multiplier: int = 3, max_positions: int = 20):
        """配置部分优先级参数
        
        Args:
            insert_multiplier: 优先级倍数，优先级N可以往前插N*multiplier个位置
            max_positions: 最大优先级插入位置限制
        """
        self.priority_insert_multiplier = insert_multiplier
        self.max_priority_positions = max_positions
        self.logger.info(f"Configured partial priority: multiplier={insert_multiplier}, max_positions={max_positions}")
        
    async def register_client(self, client_id: str, client_type: str = "unknown"):
        """注册客户端"""
        self.response_queues[client_id] = asyncio.Queue()
        self.client_stats[client_id] = {
            'total_requests': 0,
            'completed_requests': 0,
            'failed_requests': 0,
            'total_wait_time': 0,
            'client_type': client_type
        }
        self.client_request_counts[client_id] = 0
        self.client_token_stats[client_id] = {
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'actual_tokens_used': 0
        }
        self.logger.info(f"Registered client: {client_id} (type: {client_type})")
    
    async def submit_request(self, start_time: float, client_id: str, worker_id: str, request_content: str, 
                           experiment: Any, priority: int = 0) -> str:
        """提交请求到队列"""
        if client_id not in self.response_queues:
            await self.register_client(client_id)
        
        request = QueuedRequest(
            start_time=start_time,
            client_id=client_id,
            worker_id=worker_id,
            request_content=request_content,
            experiment=experiment,
            priority=priority,
            client_type=self.client_stats[client_id]['client_type']
        )
        
        # 更新客户端统计
        self.client_stats[client_id]['total_requests'] += 1
        self.client_request_counts[client_id] += 1
        
        try:
            if self.strategy == QueueStrategy.PRIORITY:
                # 部分优先级策略：根据优先级往前插入几个位置
                if priority > 0:
                    insert_positions = min(priority * self.priority_insert_multiplier, self.max_priority_positions)
                    # 确定插入位置
                    if len(self.priority_queue_list) == 0:
                        # 队列为空，直接添加
                        self.priority_queue_list.append(request)
                        insert_pos = 0
                    else:
                        # 计算插入位置：从队列末尾往前数insert_positions个位置
                        current_queue_size = len(self.priority_queue_list)
                        insert_pos = max(0, current_queue_size - insert_positions)
                        
                        # 插入到计算出的位置
                        self.priority_queue_list.insert(insert_pos, request)
                else:
                    self.priority_queue_list.append(request)
                    insert_pos = len(self.priority_queue_list) - 1
                
                self.logger.debug(f"Added request to priority queue: {client_id} (priority: {priority}, inserted at position: {insert_pos}/{len(self.priority_queue_list)})")
            else:
                # 其他策略使用普通队列
                await self.request_queue.put(request)
                self.logger.debug(f"Submitted request from {client_id} to queue")
            
            return f"request_{client_id}_{worker_id}_{int(time.time() * 1000)}"
        except asyncio.QueueFull:
            self.logger.error(f"Queue is full, rejecting request from {client_id}")
            self.client_stats[client_id]['failed_requests'] += 1
            raise Exception("Request queue is full")
    
    async def get_response(self, client_id: str, timeout: float = 30.0) -> Optional[Any]:
        """获取客户端的响应"""
        if client_id not in self.response_queues:
            return None
        
        try:
            response = await asyncio.wait_for(
                self.response_queues[client_id].get(), 
                timeout=timeout
            )
            return response
        except asyncio.TimeoutError:
            self.logger.warning(f"Response timeout after {timeout} seconds for client {client_id}")
            return None
    
    async def _get_next_request(self) -> Optional[QueuedRequest]:
        """根据策略获取下一个请求"""
        if self.strategy == QueueStrategy.FIFO:
            return await self._get_fifo_request()
        elif self.strategy == QueueStrategy.PRIORITY:
            return await self._get_priority_request()
        elif self.strategy == QueueStrategy.ROUND_ROBIN:
            return await self._get_round_robin_request()
        elif self.strategy == QueueStrategy.SHORTEST_JOB_FIRST:
            return await self._get_sjf_request()
        elif self.strategy == QueueStrategy.FAIR_SHARE:
            return await self._get_fair_share_request()
        elif self.strategy == QueueStrategy.VTC:
            return await self._get_vtc_request()
        else:
            return await self._get_fifo_request()
    
    async def _get_fifo_request(self) -> Optional[QueuedRequest]:
        """FIFO策略"""
        try:
            return await asyncio.wait_for(self.request_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return None
    
    async def _get_priority_request(self) -> Optional[QueuedRequest]:
        """部分优先级策略：从列表头部取出请求"""
        if self.priority_queue_list:
            return self.priority_queue_list.pop(0)  # 从头部取出（FIFO基础上的部分优先级）
        return None
    
    async def _get_round_robin_request(self) -> Optional[QueuedRequest]:
        """轮询策略"""
        # 简化的轮询实现，实际使用时可以更复杂
        try:
            return await asyncio.wait_for(self.request_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return None
    
    async def _get_sjf_request(self) -> Optional[QueuedRequest]:
        """最短作业优先策略"""
        # 需要收集一些请求后按estimated_tokens排序
        # 这里简化为FIFO
        try:
            return await asyncio.wait_for(self.request_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return None
    
    async def _get_fair_share_request(self) -> Optional[QueuedRequest]:
        """公平共享策略"""
        # 简化实现：优先处理请求数较少的客户端
        try:
            return await asyncio.wait_for(self.request_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return None
    
    async def _get_vtc_request(self) -> Optional[QueuedRequest]:
        """VTC策略：选择actual_tokens_used最小的客户端的最早请求"""
        if self.request_queue.qsize() == 0:
            return None
        
        # 收集所有请求
        all_requests = []
        temp_queue_size = self.request_queue.qsize()
        
        for _ in range(temp_queue_size):
            try:
                request = await asyncio.wait_for(self.request_queue.get(), timeout=0.1)
                all_requests.append(request)
            except asyncio.TimeoutError:
                break
        
        if not all_requests:
            return None
        
        # 找到tokens最少的请求
        min_tokens_request = None
        min_tokens = float('inf')
        min_submit_time = float('inf')
        
        for request in all_requests:
            client_tokens = self.client_token_stats.get(request.client_id, {}).get('actual_tokens_used', 0)
            
            # 优先选择tokens最少的，如果tokens相同则选择提交时间最早的
            if (client_tokens < min_tokens or 
                (client_tokens == min_tokens and request.submit_time < min_submit_time)):
                min_tokens = client_tokens
                min_submit_time = request.submit_time
                min_tokens_request = request
        
        # 把除了选中请求外的其他请求放回队列
        for request in all_requests:
            if request != min_tokens_request:
                await self.request_queue.put(request)
        
        if min_tokens_request:
            self.logger.debug(f"VTC selected request from {min_tokens_request.client_id} (tokens: {min_tokens})")
        
        return min_tokens_request
    
    async def _process_request(self, request: QueuedRequest, worker_name) -> Any:
        """处理单个请求"""
        if not self.openai_client:
            self.logger.error("OpenAI client not set")
            return None
        else:
            selected_client = self.openai_client[int(worker_name.split('-')[1]) % len(self.openai_client)]
        
        wait_time = time.time() - request.submit_time
        self.client_stats[request.client_id]['total_wait_time'] += wait_time
        
        try:
            # 调用原有的make_request函数
            result = await make_request(
                client=selected_client,
                experiment=request.experiment,
                request=request.request_content,
                start_time=request.start_time
            )
            
            if result is None:
                self.client_stats[request.client_id]['failed_requests'] += 1
                return None
                
            self.client_stats[request.client_id]['completed_requests'] += 1
            self.total_requests_processed += 1
            
            try:
                # 从result中提取token信息并更新统计
                # result格式: (output_tokens, elapsed_time, tokens_per_second, ttft, input_token_count, slo_compliance)
                if isinstance(result, (tuple, list)) and len(result) >= 6:  # 确保有6个返回值
                    output_tokens = int(result[0])
                    input_token_count = int(result[4])
                    
                    self.client_token_stats[request.client_id]['total_output_tokens'] += output_tokens
                    self.client_token_stats[request.client_id]['total_input_tokens'] += input_token_count
                    self.client_token_stats[request.client_id]['actual_tokens_used'] += (output_tokens + input_token_count)
            except (ValueError, TypeError, IndexError) as e:
                self.logger.error(f"Error processing result data: {str(e)}, result: {result}")
            
            return result
        except Exception as e:
            self.logger.error(f"Error processing request from {request.client_id}: {str(e)}")
            self.client_stats[request.client_id]['failed_requests'] += 1
            return None
    
    async def start_processing(self, num_workers: int = 5):
        """启动请求处理"""
        if self.is_running:
            self.logger.warning("Queue manager is already running")
            return
        
        self.is_running = True
        self.workers_running = True
        self.start_time = time.time()
        
        self.logger.info(f"Starting request queue manager with {num_workers} workers")
        
        # 创建工作协程
        workers = [
            asyncio.create_task(self._worker(f"worker-{i}"))
            for i in range(num_workers)
        ]
        
        try:
            await asyncio.gather(*workers)
        except Exception as e:
            self.logger.error(f"Error in queue processing: {e}")
        finally:
            self.is_running = False
            self.workers_running = False
    
    async def _worker(self, worker_name: str):
        """工作协程"""
        self.logger.info(f"Worker {worker_name} started")
        
        while self.workers_running:
            try:
                request = await self._get_next_request()
                if request is None:
                    await asyncio.sleep(0.1)  # 没有请求时短暂休眠
                    continue
                
                if not isinstance(request, QueuedRequest):
                    self.logger.error(f"Invalid request type: {type(request)}")
                    continue
                
                # 处理请求
                try:
                    result = await self._process_request(request, worker_name)
                    
                    # 将结果发送到客户端的响应队列
                    if request.client_id in self.response_queues:
                        await self.response_queues[request.client_id].put(result)
                except Exception as e:
                    self.logger.error(f"Error in _process_request: {str(e)}")
                    if request.client_id in self.response_queues:
                        await self.response_queues[request.client_id].put(None)
                
            except Exception as e:
                self.logger.error(f"Worker {worker_name} error: {str(e)}")
                await asyncio.sleep(1.0)
        
        self.logger.info(f"Worker {worker_name} stopped")
    
    async def stop(self):
        """停止队列管理器"""
        self.logger.info("Stopping request queue manager")
        self.workers_running = False
        self.is_running = False
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_time = time.time() - self.start_time if self.start_time else 0
        
        stats = {
            'total_requests_processed': self.total_requests_processed,
            'total_time': total_time,
            'requests_per_second': self.total_requests_processed / total_time if total_time > 0 else 0,
            'queue_size': self.request_queue.qsize(),
            'priority_queue_size': len(self.priority_queue_list),
            'client_stats': self.client_stats.copy(),
            'client_token_stats': self.client_token_stats.copy()
        }
        
        return stats
    
    def print_statistics(self):
        """打印统计信息"""
        stats = self.get_statistics()
        self.logger.info("=== Queue Manager Statistics ===")
        self.logger.info(f"Total requests processed: {stats['total_requests_processed']}")
        self.logger.info(f"Requests per second: {stats['requests_per_second']:.2f}")
        self.logger.info(f"Current queue size: {stats['queue_size']}")
        
        for client_id, client_stats in stats['client_stats'].items():
            avg_wait = client_stats['total_wait_time'] / max(client_stats['completed_requests'], 1)
            success_rate = client_stats['completed_requests'] / max(client_stats['total_requests'], 1) * 100
            
            # 获取token统计信息
            token_stats = stats['client_token_stats'].get(client_id, {})
            total_input = token_stats.get('total_input_tokens', 0)
            total_output = token_stats.get('total_output_tokens', 0)
            actual_used = token_stats.get('actual_tokens_used', 0)
            
            self.logger.info(f"Client {client_id}: {client_stats['completed_requests']}/{client_stats['total_requests']} "
                           f"(success: {success_rate:.1f}%, avg_wait: {avg_wait:.3f}s)")
            self.logger.info(f"  Tokens - Input: {total_input}, Output: {total_output}, "
                           f"Total Used: {actual_used}")
    
    async def cleanup(self):
        """清理资源"""
        self.logger.info("Cleaning up queue manager resources")
        
        # 只停止当前实验的workers
        self.workers_running = False
        
        # 清空队列
        while not self.request_queue.empty():
            try:
                self.request_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        # 清空优先级队列
        self.priority_queue_list.clear()
        
        # 重置统计信息
        self.total_requests_processed = 0
        self.start_time = None
        
        # 重置所有客户端的token统计
        for client_id in self.client_token_stats:
            self.client_token_stats[client_id] = {
                'total_input_tokens': 0,
                'total_output_tokens': 0,
                'actual_tokens_used': 0
            }
        
        self.logger.info("Queue manager cleanup completed") 
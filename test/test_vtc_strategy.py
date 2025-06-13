#!/usr/bin/env python3
"""
测试VTC策略

这个脚本测试VTC策略是否正确选择actual_tokens_used最小的客户端的第一个请求。
"""

import asyncio
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from RequestQueueManager.RequestQueueManager import RequestQueueManager, QueueStrategy

class MockExperiment:
    """模拟实验对象"""
    def __init__(self):
        self.output_tokens = 100
        self.request_timeout = 30
        self.latency_slo = 10

async def test_vtc_strategy():
    """测试VTC策略"""
    print("测试VTC策略")
    print("=" * 40)
    
    # 创建使用VTC策略的队列管理器
    queue_manager = RequestQueueManager(strategy=QueueStrategy.VTC)
    
    # 注册四个客户端
    await queue_manager.register_client("client_A", "normal")
    await queue_manager.register_client("client_B", "normal")
    await queue_manager.register_client("client_C", "normal")
    await queue_manager.register_client("client_D", "normal")  # 新增一个客户端
    
    print("注册了4个客户端: client_A, client_B, client_C, client_D")
    
    # 模拟不同的token使用量
    # client_A: 100 tokens
    # client_B: 30 tokens (最少，但不会有请求)
    # client_C: 50 tokens (第二少)
    # client_D: 200 tokens
    queue_manager.client_token_stats["client_A"]['actual_tokens_used'] = 100
    queue_manager.client_token_stats["client_B"]['actual_tokens_used'] = 30   # 最少，但没有请求
    queue_manager.client_token_stats["client_C"]['actual_tokens_used'] = 50   # 第二少
    queue_manager.client_token_stats["client_D"]['actual_tokens_used'] = 200
    
    print("\n设置token使用量:")
    print("- client_A: 100 tokens")
    print("- client_B: 30 tokens (最少，但不会有请求)")
    print("- client_C: 50 tokens (第二少)")
    print("- client_D: 200 tokens")
    
    # 提交一些请求 - 注意client_B不提交请求
    mock_experiment = MockExperiment()
    
    # client_A提交2个请求
    for i in range(2):
        await queue_manager.submit_request(
            start_time=0.0,
            client_id="client_A",
            worker_id=f"worker_A_{i}",
            request_content=f"Request from client_A #{i}",
            experiment=mock_experiment,
            priority=0
        )
    
    # client_B不提交任何请求
    
    # client_C提交2个请求
    for i in range(2):
        await queue_manager.submit_request(
            start_time=0.0,
            client_id="client_C",
            worker_id=f"worker_C_{i}",
            request_content=f"Request from client_C #{i}",
            experiment=mock_experiment,
            priority=0
        )
    
    # client_D提交1个请求
    await queue_manager.submit_request(
        start_time=0.0,
        client_id="client_D",
        worker_id="worker_D_0",
        request_content="Request from client_D #0",
        experiment=mock_experiment,
        priority=0
    )
    
    print(f"\n提交请求:")
    print(f"- client_A: 2个请求")
    print(f"- client_B: 0个请求 (虽然token最少)")
    print(f"- client_C: 2个请求")
    print(f"- client_D: 1个请求")
    print(f"队列中总共有 {queue_manager.request_queue.qsize()} 个请求")
    
    # 测试VTC策略选择
    print("\n测试VTC策略选择:")
    for round_num in range(5):  # 处理所有5个请求
        request = await queue_manager._get_vtc_request()
        if request:
            current_tokens = queue_manager.client_token_stats[request.client_id]['actual_tokens_used']
            print(f"第{round_num + 1}轮: 选择了 {request.client_id} 的请求 ({request.worker_id}), 当前tokens: {current_tokens}")
            
            # 模拟处理完成，更新token统计（假设每个请求处理后增加10个tokens）
            queue_manager.client_token_stats[request.client_id]['actual_tokens_used'] += 10
            
            # 显示更新后的状态
            print(f"      处理后 {request.client_id} tokens: {queue_manager.client_token_stats[request.client_id]['actual_tokens_used']}")
        else:
            print(f"第{round_num + 1}轮: 没有更多请求")
            break
    
    print(f"\n处理完成后的token使用量:")
    for client_id, token_stats in queue_manager.client_token_stats.items():
        print(f"- {client_id}: {token_stats['actual_tokens_used']} tokens")
    
    print(f"\n✓ VTC策略测试完成")
    print("预期结果:")
    print("1. 应该优先选择队列中有请求的客户端里token使用量最少的client_C")
    print("2. client_B虽然token最少，但没有请求，所以不会被选择")
    print("3. 随着处理进行，应该动态选择当前token最少的客户端")

if __name__ == "__main__":
    asyncio.run(test_vtc_strategy()) 
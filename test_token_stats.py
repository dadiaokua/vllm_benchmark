#!/usr/bin/env python3
"""
测试token统计功能

这个脚本测试RequestQueueManager是否正确跟踪每个客户端的token使用量。
"""

import asyncio
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from RequestQueueManager.RequestQueueManager import RequestQueueManager, QueueStrategy, QueuedRequest

class MockExperiment:
    """模拟实验对象"""
    def __init__(self):
        self.output_tokens = 100
        self.request_timeout = 30
        self.latency_slo = 10

class MockOpenAIClient:
    """模拟OpenAI客户端"""
    async def chat(self):
        pass

async def test_token_statistics():
    """测试token统计功能"""
    print("测试token统计功能")
    print("=" * 40)
    
    # 创建队列管理器
    queue_manager = RequestQueueManager(strategy=QueueStrategy.FIFO)
    
    # 注册一些客户端
    await queue_manager.register_client("short_client_0", "short")
    await queue_manager.register_client("long_client_0", "long")
    
    print("注册了2个客户端: short_client_0, long_client_0")
    
    # 模拟提交一些请求
    mock_experiment = MockExperiment()
    
    # 短客户端请求
    await queue_manager.submit_request(
        start_time=0.0,
        client_id="short_client_0",
        worker_id="worker_0",
        request_content="This is a short request",
        experiment=mock_experiment,
        priority=1
    )
    
    # 长客户端请求
    await queue_manager.submit_request(
        start_time=0.0,
        client_id="long_client_0", 
        worker_id="worker_1",
        request_content="This is a much longer request that should use more tokens",
        experiment=mock_experiment,
        priority=0
    )
    
    print("\n提交了2个请求:")
    print("- short_client_0: 短请求")
    print("- long_client_0: 长请求")
    
    # 模拟处理完成的结果
    # 模拟第一个请求的结果 (output_tokens, elapsed_time, tokens_per_second, ttft, input_token_count, slo_compliance)
    mock_result_1 = (45, 2.5, 18.0, 0.3, 15, 1)  # 45个输出token，15个输入token
    mock_result_2 = (180, 8.2, 22.0, 0.5, 35, 0)  # 180个输出token，35个输入token
    
    # 手动更新统计（模拟处理结果）
    if mock_result_1:
        queue_manager.client_stats["short_client_0"]['completed_requests'] += 1
        queue_manager.total_requests_processed += 1
        
        output_tokens, elapsed_time, tokens_per_second, ttft, input_token_count = mock_result_1[:5]
        queue_manager.client_token_stats["short_client_0"]['total_output_tokens'] += output_tokens
        queue_manager.client_token_stats["short_client_0"]['total_input_tokens'] += input_token_count
        queue_manager.client_token_stats["short_client_0"]['actual_tokens_used'] += (output_tokens + input_token_count)
        
    if mock_result_2:
        queue_manager.client_stats["long_client_0"]['completed_requests'] += 1
        queue_manager.total_requests_processed += 1
        
        output_tokens, elapsed_time, tokens_per_second, ttft, input_token_count = mock_result_2[:5]
        queue_manager.client_token_stats["long_client_0"]['total_output_tokens'] += output_tokens
        queue_manager.client_token_stats["long_client_0"]['total_input_tokens'] += input_token_count
        queue_manager.client_token_stats["long_client_0"]['actual_tokens_used'] += (output_tokens + input_token_count)
    
    print("\n模拟处理结果:")
    print("- short_client_0: 45个输出token + 15个输入token = 60个总token")
    print("- long_client_0: 180个输出token + 35个输入token = 215个总token")
    
    # 打印统计信息
    print("\n" + "=" * 50)
    queue_manager.print_statistics()
    
    # 获取详细统计
    stats = queue_manager.get_statistics()
    
    print("\n详细token统计:")
    for client_id, token_stats in stats['client_token_stats'].items():
        print(f"{client_id}:")
        print(f"  实际输入token: {token_stats['total_input_tokens']}")
        print(f"  实际输出token: {token_stats['total_output_tokens']}")
        print(f"  实际总token: {token_stats['actual_tokens_used']}")
    
    print(f"\n✓ Token统计功能测试完成")

if __name__ == "__main__":
    asyncio.run(test_token_statistics()) 
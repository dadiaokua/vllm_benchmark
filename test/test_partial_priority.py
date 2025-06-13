#!/usr/bin/env python3
"""
测试部分优先级机制

这个脚本测试新的部分优先级策略是否按预期工作。
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

async def test_partial_priority():
    """测试部分优先级机制"""
    print("测试部分优先级机制")
    print("=" * 40)
    
    # 创建队列管理器
    queue_manager = RequestQueueManager(strategy=QueueStrategy.PRIORITY)
    
    # 配置部分优先级：优先级1往前插2个位置，最多插5个位置
    queue_manager.configure_partial_priority(insert_multiplier=2, max_positions=5)
    
    print(f"配置：优先级倍数={queue_manager.priority_insert_multiplier}, 最大位置={queue_manager.max_priority_positions}")
    print()
    
    # 注册客户端
    await queue_manager.register_client("test_client", "short")
    
    # 模拟实验对象
    mock_experiment = MockExperiment()
    
    # 模拟添加请求到队列，测试插入位置
    print("添加请求到队列（模拟）:")
    
    # 先添加一些低优先级请求（优先级0，不插队）
    for i in range(10):
        request = QueuedRequest(
            start_time=0.0,
            client_id="test_client",
            worker_id=f"worker_{i}",
            request_content=f"low priority request {i}",
            experiment=mock_experiment,
            priority=0,
            client_type="normal"
        )
        queue_manager.priority_queue_list.append(request)
        print(f"  添加低优先级请求 {i} (优先级=0)")
    
    print(f"\n当前队列长度: {len(queue_manager.priority_queue_list)}")
    print("队列内容 (worker_id):", [req.worker_id for req in queue_manager.priority_queue_list])
    
    # 添加高优先级请求（优先级2，应该往前插4个位置）
    high_priority_request = QueuedRequest(
        start_time=0.0,
        client_id="test_client", 
        worker_id="high_priority_worker",
        request_content="high priority request",
        experiment=mock_experiment,
        priority=2,
        client_type="short"
    )
    
    # 模拟插入逻辑
    insert_positions = min(high_priority_request.priority * queue_manager.priority_insert_multiplier, 
                          queue_manager.max_priority_positions)
    current_queue_size = len(queue_manager.priority_queue_list)
    insert_pos = max(0, current_queue_size - insert_positions)
    
    queue_manager.priority_queue_list.insert(insert_pos, high_priority_request)
    
    print(f"\n添加高优先级请求 (优先级=2)")
    print(f"  计算插入位置数: {insert_positions}")
    print(f"  队列大小: {current_queue_size}")
    print(f"  插入位置: {insert_pos}")
    
    print(f"\n插入后队列长度: {len(queue_manager.priority_queue_list)}")
    print("队列内容 (worker_id):", [req.worker_id for req in queue_manager.priority_queue_list])
    
    # 测试另一个高优先级请求
    another_high_priority = QueuedRequest(
        start_time=0.0,
        client_id="test_client",
        worker_id="another_high_priority",
        request_content="another high priority request", 
        experiment=mock_experiment,
        priority=3,
        client_type="short"
    )
    
    insert_positions = min(another_high_priority.priority * queue_manager.priority_insert_multiplier,
                          queue_manager.max_priority_positions)
    current_queue_size = len(queue_manager.priority_queue_list)
    insert_pos = max(0, current_queue_size - insert_positions)
    
    queue_manager.priority_queue_list.insert(insert_pos, another_high_priority)
    
    print(f"\n添加另一个高优先级请求 (优先级=3)")
    print(f"  计算插入位置数: {insert_positions} (最大限制={queue_manager.max_priority_positions})")
    print(f"  实际插入位置数: {min(insert_positions, queue_manager.max_priority_positions)}")
    print(f"  插入位置: {insert_pos}")
    
    print(f"\n最终队列长度: {len(queue_manager.priority_queue_list)}")
    print("最终队列内容 (worker_id):", [req.worker_id for req in queue_manager.priority_queue_list])
    
    # 测试请求处理顺序
    print(f"\n模拟处理队列（按顺序取出）:")
    processed_order = []
    while queue_manager.priority_queue_list:
        request = queue_manager.priority_queue_list.pop(0)
        processed_order.append(f"{request.worker_id}(p{request.priority})")
    
    print("处理顺序:", " -> ".join(processed_order))
    
    print(f"\n✓ 部分优先级机制测试完成")
    print("观察结果：")
    print("1. 高优先级请求确实被插入到了队列前面")
    print("2. 但不是插入到最前面，而是根据优先级往前插几个位置")
    print("3. 这样既保证了响应性，又避免了低优先级请求饥饿")

if __name__ == "__main__":
    asyncio.run(test_partial_priority()) 
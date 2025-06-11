#!/usr/bin/env python3
"""
队列管理器使用示例

这个示例展示了如何使用新的队列管理器来控制请求顺序。
队列管理器允许所有客户端将请求提交到一个统一的队列，
然后按照指定的策略（FIFO、优先级、轮询等）来处理这些请求。
"""

import asyncio
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def print_usage():
    """打印使用说明"""
    print("队列管理器使用示例")
    print("=" * 50)
    print()
    print("1. 使用FIFO队列策略运行基准测试:")
    print("python run_benchmarks.py --exp QUEUE_FIFO --vllm_url http://localhost:8000/v1 \\")
    print("    --api_key test --short_qpm 10 --long_qpm 5 \\")
    print("    --short_clients 2 --long_clients 1 \\")
    print("    --short_clients_slo 5 --long_clients_slo 10 \\")
    print("    --short_client_qpm_ratio 1.2 --long_client_qpm_ratio 1.1 \\")
    print("    --local_port 8080 --remote_port 8080 --round 3")
    print()
    print("2. 使用优先级队列策略运行基准测试:")
    print("python run_benchmarks.py --exp QUEUE_PRIORITY --vllm_url http://localhost:8000/v1 \\")
    print("    --api_key test --short_qpm 10 --long_qpm 5 \\")
    print("    --short_clients 2 --long_clients 1 \\")
    print("    --short_clients_slo 5 --long_clients_slo 10 \\")
    print("    --short_client_qpm_ratio 1.2 --long_client_qpm_ratio 1.1 \\")
    print("    --local_port 8080 --remote_port 8080 --round 3")
    print()
    print("3. 使用轮询队列策略运行基准测试:")
    print("python run_benchmarks.py --exp QUEUE_ROUND_ROBIN --vllm_url http://localhost:8000/v1 \\")
    print("    --api_key test --short_qpm 10 --long_qpm 5 \\")
    print("    --short_clients 2 --long_clients 1 \\")
    print("    --short_clients_slo 5 --long_clients_slo 10 \\")
    print("    --short_client_qpm_ratio 1.2 --long_client_qpm_ratio 1.1 \\")
    print("    --local_port 8080 --remote_port 8080 --round 3")
    print()
    print("4. 使用最短作业优先队列策略:")
    print("python run_benchmarks.py --exp QUEUE_SJF --vllm_url http://localhost:8000/v1 \\")
    print("    --api_key test --short_qpm 10 --long_qpm 5 \\")
    print("    --short_clients 2 --long_clients 1 \\")
    print("    --short_clients_slo 5 --long_clients_slo 10 \\")
    print("    --short_client_qpm_ratio 1.2 --long_client_qpm_ratio 1.1 \\")
    print("    --local_port 8080 --remote_port 8080 --round 3")
    print()
    print("5. 使用公平共享队列策略:")
    print("python run_benchmarks.py --exp QUEUE_FAIR --vllm_url http://localhost:8000/v1 \\")
    print("    --api_key test --short_qpm 10 --long_qpm 5 \\")
    print("    --short_clients 2 --long_clients 1 \\")
    print("    --short_clients_slo 5 --long_clients_slo 10 \\")
    print("    --short_client_qpm_ratio 1.2 --long_client_qpm_ratio 1.1 \\")
    print("    --local_port 8080 --remote_port 8080 --round 3")
    print()
    print("队列策略说明:")
    print("- QUEUE_FIFO: 先进先出，按照请求到达顺序处理")
    print("- QUEUE_PRIORITY: 部分优先级队列，短请求可以往前插队，但不会完全抢占")
    print("  * 优先级1: 往前插3个位置")
    print("  * 优先级2: 往前插6个位置") 
    print("  * 优先级3: 往前插9个位置")
    print("  * 最多往前插20个位置，避免完全抢占")
    print("- QUEUE_ROUND_ROBIN: 轮询调度，公平地轮流处理各个客户端的请求")
    print("- QUEUE_SJF: 最短作业优先，估算token数量少的请求先处理")
    print("- QUEUE_FAIR: 公平共享，确保各个客户端获得公平的处理机会")
    print()
    print("队列管理器的优势:")
    print("1. 统一请求管理：所有客户端的请求都经过统一的队列")
    print("2. 可控的调度策略：可以根据需要选择不同的调度算法")
    print("3. 更好的资源利用：避免客户端直接竞争，减少资源争用")
    print("4. 详细的统计信息：提供队列统计和客户端性能指标")
    print("5. 灵活的优先级控制：部分优先级避免低优先级请求饥饿")
    print("6. 可配置的优先级参数：可以调整插队位置和最大插队数量")

async def demo_queue_manager():
    """演示队列管理器的基本功能"""
    from RequestQueueManager.RequestQueueManager import RequestQueueManager, QueueStrategy
    from util.BaseUtil import initialize_clients
    
    print("队列管理器演示")
    print("=" * 30)
    
    # 创建队列管理器
    queue_manager = RequestQueueManager(strategy=QueueStrategy.PRIORITY)
    
    # 配置部分优先级参数
    queue_manager.configure_partial_priority(insert_multiplier=4, max_positions=15)
    
    # 模拟OpenAI客户端（这里用None代替，实际使用时需要真实的客户端）
    # openai_client = initialize_clients([8080])
    # queue_manager.set_openai_client(openai_client)
    
    print(f"创建了队列管理器，策略: {queue_manager.strategy.value}")
    print(f"部分优先级配置：倍数={queue_manager.priority_insert_multiplier}, 最大位置={queue_manager.max_priority_positions}")
    
    # 注册一些客户端
    await queue_manager.register_client("short_0", "short")
    await queue_manager.register_client("short_1", "short") 
    await queue_manager.register_client("long_0", "long")
    
    print("注册了3个客户端: short_0, short_1, long_0")
    
    # 模拟提交一些请求（注意：这里不会真正发送请求，因为没有真实的OpenAI客户端）
    print("\n模拟提交请求...")
    print("优先级策略说明：")
    print("- 优先级1的请求可以往前插4个位置")
    print("- 优先级2的请求可以往前插8个位置")
    print("- 优先级3的请求可以往前插12个位置")
    print("- 最多往前插15个位置")
    
    # 获取统计信息
    stats = queue_manager.get_statistics()
    print(f"\n队列统计信息:")
    print(f"- 总处理请求数: {stats['total_requests_processed']}")
    print(f"- 当前队列大小: {stats['queue_size']}")
    print(f"- 客户端数量: {len(stats['client_stats'])}")
    
    for client_id, client_stats in stats['client_stats'].items():
        print(f"  - {client_id}: {client_stats}")

if __name__ == "__main__":
    print_usage()
    print("\n" + "=" * 80 + "\n")
    
    # 运行演示
    try:
        asyncio.run(demo_queue_manager())
    except KeyboardInterrupt:
        print("\n演示被用户中断")
    except Exception as e:
        print(f"\n演示过程中出现错误: {e}") 
import asyncio
import time
import json
from typing import Dict, List, Any
import threading

from vllm_benchmark import run_benchmark


# 结果监控器类
class BenchmarkMonitor:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.results = {}
        self.active_benchmarks = set()
        self.completed_benchmarks = set()
        self.is_monitoring = True
        self.status_updates = []

    async def register_benchmark(self, benchmark_id):
        """注册一个新的基准测试任务"""
        async with self.lock:
            self.active_benchmarks.add(benchmark_id)
            self.status_updates.append({
                "timestamp": time.time(),
                "event": "benchmark_started",
                "benchmark_id": benchmark_id
            })

    async def complete_benchmark(self, benchmark_id):
        """标记一个基准测试任务为已完成"""
        async with self.lock:
            if benchmark_id in self.active_benchmarks:
                self.active_benchmarks.remove(benchmark_id)
                self.completed_benchmarks.add(benchmark_id)
                self.status_updates.append({
                    "timestamp": time.time(),
                    "event": "benchmark_completed",
                    "benchmark_id": benchmark_id
                })

    async def add_result(self, client_type, client_index, config_index, result):
        """添加一个基准测试结果"""
        benchmark_id = f"{client_type}_{client_index}_{config_index}"

        async with self.lock:
            self.results[benchmark_id] = {
                "client_type": client_type,
                "client_index": client_index,
                "config_index": config_index,
                "timestamp": time.time(),
                "result": result
            }

            self.status_updates.append({
                "timestamp": time.time(),
                "event": "result_added",
                "benchmark_id": benchmark_id,
                "summary": self._get_result_summary(result)
            })

    def _get_result_summary(self, result):
        """获取结果摘要信息"""
        if not result:
            return {"status": "no_data"}

        summary = {}
        # 提取关键指标
        if "qps" in result:
            summary["qps"] = result["qps"]
        if "successful_requests" in result and "total_requests" in result:
            summary["success_rate"] = result["successful_requests"] / result["total_requests"] * 100
        if "avg_latency_ms" in result:
            summary["avg_latency_ms"] = result["avg_latency_ms"]
        if "avg_tps" in result:
            summary["avg_tps"] = result["avg_tps"]

        return summary

    async def get_results(self):
        """获取所有结果的副本"""
        async with self.lock:
            return self.results.copy()

    async def get_status(self):
        """获取当前状态"""
        async with self.lock:
            return {
                "active_benchmarks": list(self.active_benchmarks),
                "completed_benchmarks": list(self.completed_benchmarks),
                "total_results": len(self.results)
            }

    def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False


# 创建全局监控器实例
_BENCHMARK_MONITOR = BenchmarkMonitor()


# 监控函数
async def monitor_results(result_queue, interval=2.0):
    """监控基准测试结果并将它们发送到结果队列"""
    monitor = _BENCHMARK_MONITOR

    print("\n--- 基准测试监控已启动 ---")

    last_report_time = 0

    while monitor.is_monitoring:
        status = await monitor.get_status()
        results = await monitor.get_results()

        # 每隔一定时间打印状态报告
        current_time = time.time()
        if current_time - last_report_time >= 30:  # 每10秒打印一次状态
            last_report_time = current_time

            print(f"\n--- 监控状态更新 ({time.strftime('%H:%M:%S')}) ---")
            print(f"活跃基准测试: {len(status['active_benchmarks'])}")
            print(f"已完成基准测试: {len(status['completed_benchmarks'])}")
            print(f"总结果数: {status['total_results']}")

            # 按客户端类型分组统计
            short_results = []
            long_results = []

            for benchmark_id, data in results.items():
                if data.get("client_type") == "short":
                    short_results.append(data)
                elif data.get("client_type") == "long":
                    long_results.append(data)

            print(f"短上下文客户端结果: {len(short_results)}")
            print(f"长上下文客户端结果: {len(long_results)}")

            # 计算平均延迟和QPS
            avg_latency = 0
            avg_qps = 0
            count = 0

            for data in results.values():
                result = data.get("result", {})
                if result and "avg_latency_ms" in result:
                    avg_latency += result["avg_latency_ms"]
                    if "qps" in result:
                        avg_qps += result["qps"]
                    count += 1

            if count > 0:
                avg_latency /= count
                avg_qps /= count
                print(f"平均延迟: {avg_latency:.2f} ms")
                print(f"平均QPS: {avg_qps:.2f}")

        # 将新结果发送到队列
        for benchmark_id, data in results.items():
            # 如果结果尚未报告
            if not data.get("reported", False):
                # 添加到结果队列
                await result_queue.put(data)
                # 标记为已报告
                data["reported"] = True

        # 如果没有活跃的基准测试且至少有一个已完成的基准测试，则可以考虑退出
        if not status["active_benchmarks"] and status["completed_benchmarks"]:
            # 检查是否所有客户端都已完成
            # 这里可以添加更多的逻辑来确定是否所有预期的基准测试都已完成
            print("\n--- 所有基准测试已完成，监控退出 ---")
            # 发送None信号表示结束
            await result_queue.put(None)
            break

        await asyncio.sleep(interval)

    print("监控线程已结束")


# 装饰器函数，用于拦截原始run_benchmark的结果
async def monitor_benchmark(client_type, client_index, config_index, result):
    """将基准测试结果添加到监控器"""
    benchmark_id = f"{client_type}_{client_index}_{config_index}"

    # 注册基准测试开始
    await _BENCHMARK_MONITOR.register_benchmark(benchmark_id)

    # 添加结果
    await _BENCHMARK_MONITOR.add_result(client_type, client_index, config_index, result)

    # 标记基准测试完成
    await _BENCHMARK_MONITOR.complete_benchmark(benchmark_id)

    return result


# 装饰原始的run_benchmark函数
async def patched_run_benchmark(num_requests, concurrency, max_tokens_per_second, output_tokens,
                                clients, distribution, qps, client_index, formatted_json,
                                config_index, tokenizer):
    """
    这是一个包装函数，它调用原始的run_benchmark函数并拦截结果
    """
    # 解析client_index以获取客户端类型和索引
    parts = client_index.split("_")
    client_type = parts[0]
    index = parts[1] if len(parts) > 1 else "0"

    # 注册基准测试开始
    benchmark_id = f"{client_type}_{index}_{config_index}"
    await _BENCHMARK_MONITOR.register_benchmark(benchmark_id)

    try:
        # 调用原始的run_benchmark函数
        # 注意：这里需要替换为对您原始run_benchmark函数的实际调用
        result = await run_benchmark(num_requests, concurrency, max_tokens_per_second,
                                     output_tokens, clients, distribution, qps,
                                     client_index, formatted_json, config_index, tokenizer)
    except Exception as e:
        # 如果发生错误，记录错误信息
        result = {
            "error": str(e),
            "status": "failed",
            "qps": qps,
            "concurrency": concurrency,
            "num_requests": num_requests
        }

    # 添加结果到监控器
    await _BENCHMARK_MONITOR.add_result(client_type, index, config_index, result)

    # 标记基准测试完成
    await _BENCHMARK_MONITOR.complete_benchmark(benchmark_id)

    return result


# 设置监控
def setup_benchmark_monitoring(result_queue):
    """设置并启动基准测试监控"""
    return asyncio.create_task(monitor_results(result_queue))
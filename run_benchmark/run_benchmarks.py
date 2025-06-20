#!/usr/bin/env python3
"""
vLLM基准测试主程序
重构版本 - 只保留主干逻辑，具体功能已分离到各个模块
"""

import asyncio
import time

from config.Config import GLOBAL_CONFIG

# 导入分离的模块
from logger_utils import setup_logger
from argument_parser import parse_args, print_benchmark_config, validate_args
from engine_manager import start_vllm_engine, stop_vllm_engine
from server_manager import setup_servers_if_needed, cleanup_servers, setup_request_model_name
from result_processor import prepare_results_file, process_and_save_results
from task_manager import setup_benchmark_tasks, run_benchmark_tasks, cancel_monitor_task


async def main():
    """主函数 - 协调各个模块完成基准测试"""
    # 1. 初始化日志系统
    logger = setup_logger()
    
    # 2. 解析和验证参数
    args = parse_args(logger)
    print_benchmark_config(args, logger)
    
    args = validate_args(args, logger)
    if args is None:
        logger.error("Parameter validation failed, exiting...")
        return

    # 3. 启动vLLM引擎（如果需要）
    vllm_engine = None
    if getattr(args, 'start_engine', True):
        vllm_engine = await start_vllm_engine(args, logger)
        if vllm_engine is None:
            logger.error("Failed to start vLLM engine, exiting...")
            return
        
        # 添加vLLM引擎到全局配置，以便其他模块可以访问
        GLOBAL_CONFIG['vllm_engine'] = vllm_engine

    # 4. 设置全局配置
    try:
        GLOBAL_CONFIG['round_time'] = args.round_time
        if GLOBAL_CONFIG.get('exp_time', 36000) < args.round_time * args.round:
            GLOBAL_CONFIG['exp_time'] = args.round_time * args.round * 3
    except ImportError:
        logger.warning("Could not import GLOBAL_CONFIG, using default settings")

    # 5. 设置服务器和连接
    servers = setup_servers_if_needed(args)
    setup_request_model_name(args)
    prepare_results_file()

    # 6. 创建结果队列
    all_results = asyncio.Queue()
    request_queue = asyncio.Queue()

    start_time = time.time()

    try:
        # 7. 设置和运行基准测试任务
        logger.info("Setting up benchmark tasks...")
        tasks, monitor_task, clients, queue_manager = await setup_benchmark_tasks(
            args, all_results, request_queue, logger
        )

        # 8. 运行基准测试
        logger.info("Running benchmark tasks...")
        try:
            await run_benchmark_tasks(tasks, logger)
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")

        # 9. 处理和保存结果
        logger.info("Processing and saving results...")
        benchmark_results, total_time, filename, plot_data = process_and_save_results(
            tasks, start_time, args, logger
        )

        # 10. 清理资源
        logger.info("Cleaning up resources...")
        cleanup_servers(servers)
        await cancel_monitor_task(monitor_task, logger)

        # 停止队列管理器（如果存在）
        if queue_manager:
            await queue_manager.stop()
            logger.info("Queue manager stopped")

        # 11. 生成结果图表
        try:
            from util.plot import plot_result
            plot_result(plot_data)
        except ImportError:
            logger.warning("Could not import plot_result, skipping result plotting")

        logger.info(f"Benchmark completed successfully! Results saved to: {filename}")

    except Exception as e:
        logger.error(f"Benchmark execution failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 确保在程序结束时停止vLLM引擎
        if vllm_engine:
            stop_vllm_engine(vllm_engine, logger)
            logger.info("vLLM engine cleanup completed")


if __name__ == "__main__":
    asyncio.run(main())

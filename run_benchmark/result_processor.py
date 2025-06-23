#!/usr/bin/env python3
"""
结果处理模块
处理基准测试结果的收集、处理和保存
"""

import json
import time
import logging
from datetime import datetime

# 导入工具模块
from util.FileSaveUtil import save_benchmark_results

logger = logging.getLogger(__name__)


def process_and_save_results(tasks, start_time, args, logger):
    """处理和保存基准测试结果"""
    all_benchmark_results = []
    for task in tasks[1:]:
        if task.done() and not task.cancelled():
            try:
                result = task.result()
                if result:
                    all_benchmark_results.append(result)
            except Exception as e:
                logger.warning(f"Task result retrieval failed: {e}")

    benchmark_results = all_benchmark_results
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Total time: {total_time:.2f} seconds")

    start_datetime = datetime.fromtimestamp(start_time)
    end_datetime = datetime.fromtimestamp(end_time)
    filename = (
        f"{args.exp}_{start_datetime.strftime('%m%d_%H-%M')}_to_{end_datetime.strftime('%H-%M')}.json"
    ).replace(" ", "_").replace(":", "-").replace("/", "-")

    args_dict = vars(args)
    plot_data = {
        "filename": filename,
        "total_time": round(total_time, 2),
    }
    plot_data.update(args_dict)
    save_benchmark_results(filename, benchmark_results, plot_data, logger)
    return benchmark_results, total_time, filename, plot_data
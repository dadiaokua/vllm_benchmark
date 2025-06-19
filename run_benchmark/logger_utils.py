#!/usr/bin/env python3
"""
日志工具模块
处理日志配置和设置
"""

import os
import logging


def setup_logger():
    """设置日志记录器"""
    # 日志文件夹和文件名
    log_dir = "../log"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "run_benchmarks.log")

    # 设置全局logger
    logger = logging.getLogger("run_benchmarks")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_file, encoding="utf-8", mode="w")
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        # 控制台输出
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger 
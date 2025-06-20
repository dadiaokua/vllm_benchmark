#!/usr/bin/env python3
"""
服务器管理模块
处理服务器连接、隧道设置和管理
"""

import logging
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


def setup_servers_if_needed(args):
    """根据需要设置服务器"""
    # 这里需要导入相关的函数，根据你的实际代码结构调整
    try:
        from util.TunnelUtil import setup_vllm_servers
        
        if getattr(args, "use_tunnel", 0):
            return setup_vllm_servers(args.vllm_url, args.local_port, args.remote_port)
        return []
    except ImportError:
        logger.warning("TunnelUtil not available, skipping tunnel setup")
        return []


def cleanup_servers(servers):
    """清理服务器连接"""
    # 这里需要导入相关的函数，根据你的实际代码结构调整
    try:
        from util.TunnelUtil import stop_tunnel
        
        for server in servers:
            stop_tunnel(server)
    except ImportError:
        logger.warning("TunnelUtil not available, skipping server cleanup")


def setup_request_model_name(args):
    """设置请求模型名称"""
    # 这里需要导入GLOBAL_CONFIG，根据你的实际代码结构调整
    try:
        from config.Config import GLOBAL_CONFIG
        
        if args.request_model_name:
            GLOBAL_CONFIG['request_model_name'] = args.request_model_name
    except ImportError:
        logger.warning("Could not import GLOBAL_CONFIG, skipping model name setup") 
#!/usr/bin/env python3
"""
服务器管理模块
处理服务器连接、隧道设置和管理
"""

import logging

logger = logging.getLogger(__name__)


def setup_servers_if_needed(args):
    """根据需要设置服务器"""
    # 这里需要导入相关的函数，根据你的实际代码结构调整
    from util.tunnel import setup_vllm_servers
    
    if getattr(args, "use_tunnel", 0):
        return setup_vllm_servers(args.vllm_url, args.local_port, args.remote_port)
    return []


def cleanup_servers(servers):
    """清理服务器连接"""
    # 这里需要导入相关的函数，根据你的实际代码结构调整
    from util.tunnel import stop_tunnel
    
    for server in servers:
        stop_tunnel(server)


def setup_request_model_name(args):
    """设置请求模型名称"""
    # 这里需要导入GLOBAL_CONFIG，根据你的实际代码结构调整
    from config import GLOBAL_CONFIG
    
    if args.request_model_name:
        GLOBAL_CONFIG['request_model_name'] = args.request_model_name 
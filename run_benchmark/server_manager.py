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

# 导入配置模块
from config.Config import GLOBAL_CONFIG

# 尝试导入隧道工具
try:
    from util.TunnelUtil import setup_vllm_servers, stop_tunnel
    tunnel_available = True
except ImportError:
    tunnel_available = False

logger = logging.getLogger(__name__)


def setup_servers(args):
    """根据需要设置服务器"""
    # 设置请求模型名称
    if args.request_model_name:
        GLOBAL_CONFIG['request_model_name'] = args.request_model_name
    
    # 设置隧道（如果需要）
    if tunnel_available and getattr(args, "use_tunnel", 0):
        return setup_vllm_servers(args.vllm_url, args.local_port, args.remote_port)
    
    if not tunnel_available:
        logger.warning("TunnelUtil not available, skipping tunnel setup")
    
    return []


def cleanup_servers(servers):
    """清理服务器连接"""
    if tunnel_available:
        for server in servers:
            stop_tunnel(server)
    else:
        logger.warning("TunnelUtil not available, skipping server cleanup") 
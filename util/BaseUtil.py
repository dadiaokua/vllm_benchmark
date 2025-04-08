from datetime import datetime
import time
import json
import os
import numpy as np
from openai import AsyncOpenAI
from sympy import symbols, Eq, solve

from config.Config import GLOBAL_CONFIG
from util.FileSaveUtil import save_exchange_record


def initialize_clients(local_port):
    """Initialize OpenAI clients based on port configuration"""
    if isinstance(local_port, list):
        return [AsyncOpenAI(base_url=f"http://localhost:{port}/v1") for port in local_port]
    else:
        return [AsyncOpenAI(base_url=f"http://localhost:{local_port}/v1")]


def ExchangeResources(client1, client2, fairness_ratio):
    print(f"[ExchangeQPS] Starting QPS exchange between client {client1.client_id} and {client2.client_id}")

    if client1.exchange_Resources_Times >= 3:
        print(f"[ExchangeQPS] Client {client1.client_id} has exchanged QPS more than 3 times, skipping")
        return

    delta = fairness_ratio * GLOBAL_CONFIG["ADJUST_SENSITIVITY"]
    tmp = (client1.service * delta / client2.service) * client2.results[-1]['total_requests']
    client2.qps = int(client2.qps + tmp)
    client1.active_ratio -= delta
    client1.credit += delta * 100
    client2.credit -= delta * 100

    client1.exchange_Resources_Times += 1

    formatted_time = datetime.now().strftime("%m_%d_%H_%M")
    # 记录资源交换信息
    exchange_record = {
        "timestamp": formatted_time,
        "client1_id": client1.client_id,
        "client2_id": client2.client_id,
        "fairness_ratio": fairness_ratio,
        "delta": delta,
        "qps_change": int(tmp),
        "client1_new_active_ratio": client1.active_ratio,
        "client1_new_credit": client1.credit,
        "client2_new_credit": client2.credit,
        "client2_new_qps": client2.qps
    }
    
    # 保存交换记录到文件
    save_exchange_record(exchange_record, f'tmp_result/resources_exchanges_{formatted_time}.json')


def selectClients(clients):
    """
    从客户端列表中选择两个客户端进行资源交换

    选择逻辑:
    1. 从两端向中间查找，找到第一对满足差值条件的客户端索引 (i, j)
    2. 在低端 (0 到 i) 选择最优的低 fairness_ratio 客户端
    3. 在高端 (j 到 end) 选择最优的高 fairness_ratio 客户端

    Args:
        clients: 已按 fairness_ratio 排序的客户端列表

    Returns:
        tuple: (低 fairness_ratio 的客户端, 高 fairness_ratio 的客户端)
    """
    n = len(clients)

    # 从两端向中间查找，找到第一对满足条件的索引
    i, j = 0, n - 1
    found = False

    while i < j:
        if abs(clients[i].fairness_ratio - clients[j].fairness_ratio) > GLOBAL_CONFIG['fairness_ratio']:
            found = True
            break

        # 移动差距较小的一端
        if clients[i + 1].fairness_ratio - clients[i].fairness_ratio < clients[j].fairness_ratio - clients[
            j - 1].fairness_ratio:
            i += 1
        else:
            j -= 1

    if not found:
        print("[Fairness] No client pairs found with sufficient fairness ratio difference")
        return clients[0], clients[-1]  # 如果没找到，返回两端的客户端

    # 在低端选择最优的客户端 (exchange_times 少且 credit 小)
    best_low = clients[0]
    for k in range(1, i + 1):
        if clients[k].exchange_Resources_Times < best_low.exchange_Resources_Times:
            best_low = clients[k]
        elif clients[k].exchange_Resources_Times == best_low.exchange_Resources_Times and clients[
            k].credit < best_low.credit:
            best_low = clients[k]

    # 在高端选择最优的客户端 (exchange_times 少且 credit 大)
    best_high = clients[n - 1]
    for k in range(n - 2, j - 1, -1):
        if clients[k].exchange_Resources_Times < best_high.exchange_Resources_Times:
            best_high = clients[k]
        elif clients[k].exchange_Resources_Times == best_high.exchange_Resources_Times and clients[
            k].credit > best_high.credit:
            best_high = clients[k]

    print(
        f"[Fairness] Selected clients with fairness ratios: {best_low.fairness_ratio:.3f} and {best_high.fairness_ratio:.3f}")
    print(f"[Fairness] Exchange times: {best_low.exchange_Resources_Times} and {best_high.exchange_Resources_Times}")
    print(f"[Fairness] Credits: {best_low.credit} and {best_high.credit}")

    return best_low, best_high

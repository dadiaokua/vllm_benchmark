import json
from datetime import datetime
from openai import AsyncOpenAI

from config.Config import GLOBAL_CONFIG
from util.FileSaveUtil import save_exchange_record


def initialize_clients(local_port):
    """Initialize OpenAI clients based on port configuration"""
    if isinstance(local_port, list):
        print(f"Initializing multiple OpenAI clients:")
        clients = []
        for port in local_port:
            url = f"http://localhost:{port}/v1"
            print(f"Creating client with base_url: {url}")
            clients.append(AsyncOpenAI(base_url=url, api_key="empty"))
        return clients
    else:
        url = f"http://localhost:{local_port}/v1"
        print(f"Initializing single OpenAI client with base_url: {url}")
        return [AsyncOpenAI(base_url=url, api_key="empty")]


def exchange_resources(client_low_fairness_ratio, client_high_fairness_ratio, clients, exp_type):
    """
    在两个客户端之间交换资源以提高公平性
    
    Args:
        client_low_fairness_ratio: fairness_ratio 较低的客户端
        client_high_fairness_ratio: fairness_ratio 较高的客户端
        clients: 所有客户端列表，用于计算平均成功率
        exp_type: 实验类型
    """
    # 1. 计算调整量
    if exp_type == "LFS":
        delta = calculate_adjustment_delta_lfs(client_low_fairness_ratio, client_high_fairness_ratio)
    elif exp_type == "VTC" or exp_type == "DLPM":
        delta = calculate_adjustment_delta_vtc(client_low_fairness_ratio, client_high_fairness_ratio)
    else:
        print(f"Invalid experiment type: {exp_type}")
        return

    # 3. 获取系统平均成功率
    avg_success_rate = get_average_success_rate(clients)
    if avg_success_rate is None:
        return  # 没有足够数据进行调整

    # 4. 根据系统负载调整资源
    adjust_resources(client_low_fairness_ratio, client_high_fairness_ratio, delta, avg_success_rate)

    # 5. 更新信用值和交换次数
    update_credits_and_counters(client_low_fairness_ratio, client_high_fairness_ratio, delta)

    # 准备客户端信息列表
    clients_info = []
    for client in clients:
        client_info = {
            "client_id": client.client_id if hasattr(client, 'client_id') else str(client),
            "fairness_ratio": client.fairness_ratio,
            "service": client.service,
            "credit": client.credit,
            "latency_slo": client.latency_slo if hasattr(client, 'latency_slo') else 0,
            "exchange_Resources_Times": client.exchange_Resources_Times
        }
        clients_info.append(client_info)

    # 准备交换记录
    exchange_record = {
        "timestamp": datetime.now().strftime("%m_%d_%H_%M_%S"),
        "client1_id": client_low_fairness_ratio.client_id if hasattr(client_low_fairness_ratio, 'client_id') else str(client_low_fairness_ratio),
        "client2_id": client_high_fairness_ratio.client_id if hasattr(client_high_fairness_ratio, 'client_id') else str(client_high_fairness_ratio),
        "gap_fairness_ratio": f"abs({client_low_fairness_ratio.fairness_ratio} - {client_high_fairness_ratio.fairness_ratio}) = {abs(client_low_fairness_ratio.fairness_ratio - client_high_fairness_ratio.fairness_ratio)}",
        "delta": delta,
        "client1_new_active_ratio": client_low_fairness_ratio.active_ratio,
        "client1_new_credit": client_low_fairness_ratio.credit,
        "client2_new_credit": client_high_fairness_ratio.credit,
        "clients_info": clients_info
    }

    save_exchange_record(exchange_record,
                         f'tmp_result/{exp_type}_resources_exchanges_{GLOBAL_CONFIG["monitor_file_time"]}.json')


def calculate_adjustment_delta_lfs(client1, client2):
    """计算调整量"""
    fairness_diff = abs(client1.fairness_ratio - client2.fairness_ratio)
    delta = fairness_diff * GLOBAL_CONFIG["ADJUST_SENSITIVITY"]
    max_delta = GLOBAL_CONFIG.get("MAX_ADJUST_DELTA", 0.5)
    return min(delta, max_delta)


def calculate_adjustment_delta_vtc(client1, client2):
    """计算调整量"""
    fairness_diff = abs(client1.service - client2.service) / max(client1.service, client2.service)
    delta = fairness_diff * GLOBAL_CONFIG.get("ADJUST_SENSITIVITY", 1)
    max_delta = GLOBAL_CONFIG.get("MAX_ADJUST_DELTA", 0.5)
    return min(delta, max_delta)


def get_average_success_rate(clients):
    """计算系统平均成功率"""
    success_rates = []
    for client in clients:
        if client.results:
            rate = client.results[-1]['successful_requests'] / client.results[-1]['total_requests']
            success_rates.append(rate)

    if not success_rates:
        print("No clients have successful requests")
        return None

    avg_rate = sum(success_rates) / len(success_rates)
    print(f"System average success rate: {avg_rate:.2f}")
    return avg_rate


def adjust_resources(client_low_fairness_ratio, client_high_fairness_ratio, delta, avg_success_rate):
    """统一的资源调整策略"""
    # 调整 time_ratio - 无论负载如何，都减少高公平性客户端的 time_ratio
    client_high_fairness_ratio.time_ratio = client_high_fairness_ratio.time_ratio * (1 - delta)
    client_low_fairness_ratio.time_ratio = client_low_fairness_ratio.time_ratio * (1 + delta)

    # 减少 client1 的 active_ratio - 无论负载如何
    min_active_ratio = GLOBAL_CONFIG.get("MIN_ACTIVE_RATIO", 0.1)
    client_low_fairness_ratio.active_ratio = max(client_low_fairness_ratio.active_ratio - delta, min_active_ratio)
    client_high_fairness_ratio.active_ratio = min(client_high_fairness_ratio.active_ratio + delta, 1)

    # 记录调整信息
    print(f"Adjusted resources: client1.time_ratio={client_low_fairness_ratio.time_ratio:.2f}, "
          f"client2.time_ratio={client_high_fairness_ratio.time_ratio:.2f}, "
          f"client1.active_ratio={client_low_fairness_ratio.active_ratio:.2f}")


def update_credits_and_counters(client1, client2, delta):
    """更新信用值和交换次数"""
    credit_change = int(delta * 10)
    client1.credit += credit_change
    client2.credit -= credit_change

    client1.exchange_Resources_Times += 1
    client2.exchange_Resources_Times += 1


def selectClients_LFS(clients):
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
        if clients[i].exchange_Resources_Times >= GLOBAL_CONFIG.get('max_exchange_times', 3):
            i += 1
            continue
        elif clients[j].exchange_Resources_Times >= GLOBAL_CONFIG.get('max_exchange_times', 3):
            j -= 1
            continue

        if abs(clients[i].fairness_ratio - clients[j].fairness_ratio) > GLOBAL_CONFIG['fairness_ratio_LFS']:
            found = True
        else:
            break
        # 移动差距较小的一端
        if clients[i + 1].fairness_ratio - clients[i].fairness_ratio < clients[j].fairness_ratio - clients[
            j - 1].fairness_ratio:
            i += 1
        else:
            j -= 1

    if not found:
        print("[Fairness] No client pairs found with sufficient fairness ratio difference")
        return None, None

    # 在低端选择最优的客户端
    best_low = clients[0]

    for k in range(1, i + 1):
        if clients[k].exchange_Resources_Times >= GLOBAL_CONFIG.get('max_exchange_times', 3):
            continue
        if clients[k].exchange_Resources_Times < best_low.exchange_Resources_Times:
            best_low = clients[k]
        elif clients[k].exchange_Resources_Times == best_low.exchange_Resources_Times and clients[
            k].credit < best_low.credit:
            best_low = clients[k]

    # 在高端选择最优的客户端
    best_high = clients[n - 1]

    for k in range(n - 2, j - 1, -1):
        if clients[k].exchange_Resources_Times >= GLOBAL_CONFIG.get('max_exchange_times', 3):
            continue
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


def selectClients_VTC(clients):
    i, j = 0, len(clients) - 1

    while i < j:
        if clients[i].exchange_Resources_Times >= GLOBAL_CONFIG.get('max_exchange_times', 3):
            i += 1
        if clients[j].exchange_Resources_Times >= GLOBAL_CONFIG.get('max_exchange_times', 3):
            j -= 1

        if abs(clients[i].service / clients[j].service) <= GLOBAL_CONFIG.get('fairness_ratio_VTC', 0.5):
            return clients[j], clients[i]
        else:
            return None, None

import json
from datetime import datetime
from openai import AsyncOpenAI

from config.Config import GLOBAL_CONFIG
from util.FileSaveUtil import save_exchange_record


def initialize_clients(local_port):
    """Initialize OpenAI clients based on port configuration"""
    if isinstance(local_port, list):
        return [AsyncOpenAI(base_url=f"http://localhost:{port}/v1") for port in local_port]
    else:
        return [AsyncOpenAI(base_url=f"http://localhost:{local_port}/v1")]


def exchange_resources(client1, client2, clients, exp_type):
    """
    在两个客户端之间交换资源以提高公平性
    
    Args:
        client1: fairness_ratio 较低的客户端
        client2: fairness_ratio 较高的客户端
        clients: 所有客户端列表，用于计算平均成功率
        exp_type: 实验类型
    """
    # 1. 计算调整量
    delta = calculate_adjustment_delta(client1, client2)

    # 2. 计算服务比例和 QPS 调整量
    service_ratio = client1.fairness_ratio / client2.fairness_ratio if client2.fairness_ratio > 0 else 1.0
    qps_adjust = calculate_qps_adjustment(client2)

    # 3. 获取系统平均成功率
    avg_success_rate = get_average_success_rate(clients)
    if avg_success_rate is None:
        return  # 没有足够数据进行调整

    # 4. 根据系统负载调整资源
    if avg_success_rate > GLOBAL_CONFIG.get('avg_success_rate', 0.9):
        adjust_for_light_load(client1, client2, qps_adjust, delta, service_ratio)
    else:
        adjust_for_heavy_load(client1, client2, qps_adjust, delta, service_ratio)

    # 5. 更新信用值和交换次数
    update_credits_and_counters(client1, client2, qps_adjust)

    # 6. 记录调整结果
    log_adjustment_results(client1, client2)

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
        "timestamp": GLOBAL_CONFIG['monitor_file_time'],
        "client1_id": client1.client_id if hasattr(client1, 'client_id') else str(client1),
        "client2_id": client2.client_id if hasattr(client2, 'client_id') else str(client2),
        "gap_fairness_ratio": f"abs({client1.fairness_ratio} - {client2.fairness_ratio}) = {abs(client1.fairness_ratio - client2.fairness_ratio)}",
        "delta": delta,
        "qps_change": f"int({qps_adjust} * ({client1.fairness_ratio}/{client2.fairness_ratio if client2.fairness_ratio > 0 else 1.0})) = {int(qps_adjust * (client1.fairness_ratio / client2.fairness_ratio if client2.fairness_ratio > 0 else 1.0))}",
        "client1_new_active_ratio": client1.active_ratio,
        "client1_new_credit": client1.credit,
        "client2_new_credit": client2.credit,
        "client2_new_qps": client2.qps,
        "clients_info": clients_info
    }

    save_exchange_record(exchange_record,
                         f'tmp_result/{exp_type}_resources_exchanges_{GLOBAL_CONFIG["monitor_file_time"]}.json')


def calculate_adjustment_delta(client1, client2):
    """计算调整量"""
    fairness_diff = abs(client1.fairness_ratio - client2.fairness_ratio)
    delta = fairness_diff * GLOBAL_CONFIG["ADJUST_SENSITIVITY"]
    max_delta = GLOBAL_CONFIG.get("MAX_ADJUST_DELTA", 0.2)
    return min(delta, max_delta)


def calculate_qps_adjustment(client):
    """计算 QPS 调整量"""
    recent_count = min(3, len(client.results))
    if recent_count == 0:
        print("Not enough results for adjustment")
        return 0

    avg_requests = sum(r['total_requests'] for r in client.results[-recent_count:]) / recent_count
    adjust_qps = max(avg_requests, 1)
    print(f"adjust_qps: {adjust_qps} = max(avg_requests, 1)")
    return adjust_qps


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


def adjust_for_light_load(client1, client2, qps_adjust, delta, service_ratio):
    """系统负载较轻时的调整策略"""
    print("System lightly loaded, increasing total QPS")

    # 增加 client2 的 QPS
    new_qps2 = int(client2.qps) + max(int(qps_adjust * service_ratio / GLOBAL_CONFIG["round_time"]), 1)
    client2.qps = min(new_qps2, GLOBAL_CONFIG.get("MAX_QPS", 1000))

    # 减少 client1 的 active_ratio
    min_active_ratio = GLOBAL_CONFIG.get("MIN_ACTIVE_RATIO", 0.1)
    client1.active_ratio = max(client1.active_ratio - delta, min_active_ratio)


def adjust_for_heavy_load(client1, client2, qps_adjust, delta, service_ratio):
    """系统负载较重时的调整策略"""
    print("System heavily loaded, redistributing QPS")

    # 增加 client2 的 QPS
    new_qps2 = int(client2.qps) + max(int(qps_adjust * service_ratio / GLOBAL_CONFIG["round_time"]), 1)
    client2.qps = min(new_qps2, GLOBAL_CONFIG.get("MAX_QPS", 1000))

    # 减少 client1 的 QPS
    new_qps1 = int(client1.qps) - max(int(qps_adjust * service_ratio / GLOBAL_CONFIG["round_time"]), 1)
    min_qps = GLOBAL_CONFIG.get("MIN_QPS", 1)
    client1.qps = max(new_qps1, min_qps)

    # 减少 client1 的 active_ratio
    min_active_ratio = GLOBAL_CONFIG.get("MIN_ACTIVE_RATIO", 0.1)
    client1.active_ratio = max(client1.active_ratio - delta, min_active_ratio)


def update_credits_and_counters(client1, client2, qps_adjust):
    """更新信用值和交换次数"""
    credit_change = int(qps_adjust * 10)
    client1.credit += credit_change
    client2.credit -= credit_change

    client1.exchange_Resources_Times += 1
    client2.exchange_Resources_Times += 1


def log_adjustment_results(client1, client2):
    """记录调整结果"""
    print(f"Resource exchange complete:")
    print(f"  Client1: QPS={client1.qps}, active_ratio={client1.active_ratio:.2f}, credit={client1.credit}")
    print(f"  Client2: QPS={client2.qps}, active_ratio={client2.active_ratio:.2f}, credit={client2.credit}")


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

    # 记录搜索过程
    log_data = {
        "timestamp": GLOBAL_CONFIG['monitor_file_time'],
        "total_clients": n,
        "search_process": []
    }

    # 从两端向中间查找，找到第一对满足条件的索引
    i, j = 0, n - 1
    found = False

    while i < j:
        # 记录每一步的搜索状态
        step_data = {
            "step": len(log_data["search_process"]) + 1,
            "i": i,
            "j": j,
            "fairness_ratio_i": clients[i].fairness_ratio,
            "fairness_ratio_j": clients[j].fairness_ratio,
            "difference": abs(clients[i].fairness_ratio - clients[j].fairness_ratio)
        }
        if clients[i].exchange_Resources_Times >= GLOBAL_CONFIG.get('max_exchange_times', 3):
            i += 1
            continue
        elif clients[j].exchange_Resources_Times >= GLOBAL_CONFIG.get('max_exchange_times', 3):
            j -= 1
            continue

        if abs(clients[i].fairness_ratio - clients[j].fairness_ratio) > GLOBAL_CONFIG['fairness_ratio_LFS']:
            found = True
            step_data["result"] = "Found matching pair"
            log_data["search_process"].append(step_data)
            break

        # 移动差距较小的一端
        if clients[i + 1].fairness_ratio - clients[i].fairness_ratio < clients[j].fairness_ratio - clients[
            j - 1].fairness_ratio:
            step_data["action"] = "Moving i forward"
            i += 1
        else:
            step_data["action"] = "Moving j backward"
            j -= 1

        log_data["search_process"].append(step_data)

    if not found:
        log_data["result"] = "No suitable pairs found"
        with open('lfs_selection.log', 'a') as f:
            json.dump(log_data, f, indent=2)
            f.write('\n')
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

    # 写入日志文件
    # save_exchange_record(log_data, f'tmp_result/lfs_selection_{GLOBAL_CONFIG["monitor_file_time"]}.log')

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

        if abs(clients[i].fairness_ratio / clients[j].fairness_ratio) < GLOBAL_CONFIG.get('fairness_ratio_VTC', 0.5):
            return clients[j], clients[i]
        else:
            return None, None

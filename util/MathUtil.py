import numpy as np

from config.Config import GLOBAL_CONFIG
from util.BaseUtil import ExchangeResources, selectClients


def calculate_Jains_index(service_list):
    service = [entry["service"] for entry in service_list]
    n = len(service)
    if n == 0:
        return 0  # Avoid division by zero

    sum_service = sum(service)
    sum_squares = sum(s ** 2 for s in service)

    j = (sum_service ** 2) / (n * sum_squares)
    return j


def calculate_service_value(total_input_tokens, total_output_tokens):
    """Calculate service value based on input and output tokens"""
    return total_input_tokens + 2 * total_output_tokens


async def fairness_result(clients):
    # Calculate service values and max service in one pass
    max_service = 0
    service = []
    
    for client in clients:
        # Get latest results
        latest_result = client.results[-1]
        
        # Calculate service value
        service_value = calculate_service_value(
            latest_result["total_input_tokens"],
            latest_result["total_output_tokens"]
        )
        
        # Update max service
        max_service = max(max_service, service_value)
        
        # Store service info
        service.append({
            "service": service_value,
            "client": latest_result["client_index"]
        })
        
        # Update client attributes
        client.service = service_value
        client.service_div_latency = service_value / client.avg_latency_div_standard_latency

    # Calculate fairness ratios in one pass
    alpha = GLOBAL_CONFIG['alpha']
    for client in clients:
        client.fairness_ratio = (1 - client.service / max_service) * (1 - alpha) + alpha * client.avg_latency_div_standard_latency

    # Calculate Jain's fairness index
    tmp_jains_index = calculate_Jains_index(service)
    
    return tmp_jains_index, service


def fairness_score(client, max_service):
    latency_ratio = client.avg_latency_div_standard_latency or 1e-6  # 避免除0
    service_ratio = client.service / max_service if max_service != 0 else 0
    return GLOBAL_CONFIG['alpha'] * latency_ratio + (1 - GLOBAL_CONFIG['alpha']) * (1 - service_ratio)


async def is_fairness(clients):
    if len(clients) < 2:
        print("[Fairness] Not enough clients for fairness calculation (minimum 3 required)")
        return
    iteration = 0

    while iteration < (len(clients) - 1):
        print(f"[Fairness] Starting iteration {iteration + 1}/{len(clients) - 1}")

        clients.sort(key=lambda client: client.fairness_ratio)

        client1, client2 = selectClients(clients)
        if client1 is not None and client2 is not None:
            fairness_ratio = abs(client1.fairness_ratio - client2.fairness_ratio)
            if fairness_ratio <= GLOBAL_CONFIG["fairness_ratio"]:
                print("[Fairness] Target fairness ratio achieved, stopping adjustments")
                return
            ExchangeResources(client1, client2, fairness_ratio)
        else:
            break

        iteration += 1

    print("[Fairness] WARNING: Reached maximum iterations without achieving target fairness ratio")


def calculate_percentile(values, percentile, reverse=False):
    """Calculate percentile value from a list"""
    if not values:
        return None
    target_percentile = 100 - percentile if reverse else percentile
    return np.percentile(values, target_percentile)


def calculate_metrics(concurrency, request_timeout, client_id, results, start_time, end_time, num_requests, qps,
                      output_tokens, latency_slo):
    # Calculate metrics
    total_elapsed_time = end_time - start_time
    total_tokens = sum(tokens for tokens, _, _, _, _, _ in results if tokens is not None)
    total_input_tokens = sum(input_token for _, _, _, _, input_token, _ in results if input_token is not None)
    latencies = [elapsed_time for _, elapsed_time, _, _, _, _ in results if elapsed_time is not None]
    tokens_per_second_list = [tps for _, _, tps, _, _, _ in results if tps is not None]
    ttft_list = [ttft for _, _, _, ttft, _, _ in results if ttft is not None]
    slo_violation_count = len([slo for _, _, _, _, _, slo in results if slo == 0])
    avg_latency_div_standard_latency = sum(latencies) / len(latencies) / (latency_slo if latency_slo > 0 else 0)

    successful_requests = len(results)
    requests_per_second = successful_requests / total_elapsed_time if total_elapsed_time > 0 else 0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    avg_tokens_per_second = sum(tokens_per_second_list) / len(
        tokens_per_second_list) if tokens_per_second_list else 0
    avg_ttft = sum(ttft_list) / len(ttft_list) if ttft_list else 0

    # Calculate percentiles
    percentiles = [50, 95, 99]
    latency_percentiles = [calculate_percentile(latencies, p) for p in percentiles]
    tps_percentiles = [calculate_percentile(tokens_per_second_list, p, reverse=True) for p in percentiles]
    ttft_percentiles = [calculate_percentile(ttft_list, p) for p in percentiles]

    return {
        "slo_violation_count": slo_violation_count,
        "avg_latency_div_standard_latency": avg_latency_div_standard_latency,
        "time": end_time,
        "qps": qps,
        "total_requests": num_requests,
        "successful_requests": successful_requests,
        "concurrency": concurrency,
        "request_timeout": request_timeout,
        "max_output_tokens": output_tokens,
        "total_time": total_elapsed_time,
        "requests_per_second": requests_per_second,
        "total_output_tokens": total_tokens,
        "total_input_tokens": total_input_tokens,
        "latency": {
            "average": avg_latency,
            "p50": latency_percentiles[0],
            "p95": latency_percentiles[1],
            "p99": latency_percentiles[2]
        },
        "tokens_per_second": {
            "average": avg_tokens_per_second,
            "p50": tps_percentiles[0],
            "p95": tps_percentiles[1],
            "p99": tps_percentiles[2]
        },
        "time_to_first_token": {
            "average": avg_ttft,
            "p50": ttft_percentiles[0],
            "p95": ttft_percentiles[1],
            "p99": ttft_percentiles[2]
        },
        "client_index": client_id,
    }

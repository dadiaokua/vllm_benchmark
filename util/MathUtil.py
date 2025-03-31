import numpy as np

from config.Config import GLOBAL_CONFIG
from util.BaseUtil import ExchangeQPS


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
    service = []
    for client in clients:
        service_value = calculate_service_value(
            client.results["total_input_tokens"],
            client.results["total_output_tokens"]
        )
        service.append({
            "service": service_value,
            "client": client.results["client_index"]
        })
        client.service = service_value

    tmp_jains_index = calculate_Jains_index(service)
    return tmp_jains_index, service


async def is_fairness(clients):
    if len(clients) <= 2:
        print("No fairness for less 2 clients")
        return
    iteration = 0

    while iteration < (len(clients) - 1):
        clients.sort(key=lambda client: client.service / client.avg_latency_div_standard_latency
        if client.avg_latency_div_standard_latency != 0 else float('inf'))

        fairness_ratio = (clients[0].service / clients[0].avg_latency_div_standard_latency) / (
                clients[-1].service / clients[-1].avg_latency_div_standard_latency)

        if fairness_ratio <= GLOBAL_CONFIG["a"]:
            print("All clients have fairness")
            return

        ExchangeQPS(clients[0], clients[-1])

        iteration += 1
        print(f"Iteration {iteration}: Adjusted QPS")

    print("Reached maximum iterations without achieving fairness")


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

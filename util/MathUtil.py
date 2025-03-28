import numpy as np


def calculate_Jains_index(service_list):
    service = [entry["service"] for entry in service_list]
    n = len(service)
    if n == 0:
        return 0  # Avoid division by zero

    sum_service = sum(service)
    sum_squares = sum(s ** 2 for s in service)

    j = (sum_service ** 2) / (n * sum_squares)
    return j


async def fairness_result(clients):
    service = []
    for client in clients:
        service.append({"service": client.results["total_input_tokens"] + 2 * client.results["total_output_tokens"],
                        "client": client.results["client_index"]})
    tmp_jains_index = calculate_Jains_index(service)
    return tmp_jains_index, service

def calculate_percentile(values, percentile, reverse=False):
    if not values:
        return None
    if reverse:
        return np.percentile(values, 100 - percentile)
    return np.percentile(values, percentile)
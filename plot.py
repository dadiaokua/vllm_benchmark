import json
import matplotlib.pyplot as plt

def plot():
    with open('benchmark_results.json', 'r') as f:
        all_results = json.load(f)
    print(all_results)
    request_number = []
    total_time = []
    total_output_tokens = []
    latency = []
    tokens_per_second = []
    time_to_first_token = []
    success_rate = []
    requests_per_second = []
    concurrency = []
    for i in range(1, len(all_results)):
        success_rate.append(all_results[i]['successful_requests']*100/all_results[i]['total_requests'])
        request_number.append(all_results[i]["total_requests"])
        total_time.append(all_results[i]["total_time"])
        total_output_tokens.append(all_results[i]["total_output_tokens"])
        latency.append(all_results[i]["latency"]["average"])
        tokens_per_second.append(all_results[i]["tokens_per_second"]["average"])
        time_to_first_token.append(all_results[i]["time_to_first_token"]["average"])
        requests_per_second.append(all_results[i]["requests_per_second"])
        concurrency.append(all_results[i]["concurrency"])

    plt.plot(concurrency, total_time, label='total_time', color='blue', marker='o')
    # plt.plot(concurrency, total_output_tokens, label='total_output_tokens', color='orange', marker='s')
    plt.plot(concurrency, latency, label='latency', color='green', marker='^')
    plt.plot(concurrency, tokens_per_second, label='tokens_per_second', color='red', marker='x')
    plt.plot(concurrency, time_to_first_token, label='time_to_first_token', color='purple', marker='*')
    plt.plot(concurrency, success_rate, label='success_rate / %', color='pink', marker='s')
    plt.plot(concurrency, requests_per_second, label='requests_per_second',color='black', marker='1')
    plt.title('test')
    plt.xlabel('concurrency')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    plot()

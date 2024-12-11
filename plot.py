import json
import math

import matplotlib.pyplot as plt

def plot():
    with open('benchmark_results.json', 'r') as f:
        all_results = json.load(f)
    print(all_results)
    if len(all_results) >= 1:
        cols = math.ceil(math.sqrt(len(all_results)))
        rows = math.ceil(len(all_results) / cols)

        # 创建子图
        fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))

        # 将 axs 转为一维数组，便于索引
        axs = axs.flatten()

        for index, all_result in enumerate(all_results):
            request_number = []
            total_time = []
            total_output_tokens = []
            latency = []
            tokens_per_second = []
            time_to_first_token = []
            success_rate = []
            requests_per_second = []
            concurrency = []
            for i in range(len(all_result)):
                success_rate.append(all_result[i]['successful_requests']*100/all_result[i]['total_requests'])
                request_number.append(all_result[i]["total_requests"])
                total_time.append(all_result[i]["total_time"])
                total_output_tokens.append(all_result[i]["total_output_tokens"])
                latency.append(all_result[i]["latency"]["average"])
                tokens_per_second.append(all_result[i]["tokens_per_second"]["average"])
                time_to_first_token.append(all_result[i]["time_to_first_token"]["average"])
                requests_per_second.append(all_result[i]["requests_per_second"])
                concurrency.append(all_result[i]["concurrency"])

            axs[index].plot(concurrency, total_time, label='total_time', color='blue', marker='o')
            # axs[index].plot(concurrency, total_output_tokens, label='total_output_tokens', color='orange', marker='s')
            axs[index].plot(concurrency, latency, label='latency', color='green', marker='^')
            axs[index].plot(concurrency, tokens_per_second, label='tokens_per_second', color='red', marker='x')
            axs[index].plot(concurrency, time_to_first_token, label='time_to_first_token', color='purple', marker='*')
            axs[index].plot(concurrency, success_rate, label='success_rate / %', color='pink', marker='s')
            axs[index].plot(concurrency, requests_per_second, label='requests_per_second',color='black', marker='1')
            axs[index].set_title(f"Client {index + 1}")
            axs[index].set_xlabel('concurrency')
            axs[index].legend()

        for j in range(len(all_results), len(axs)):
            fig.delaxes(axs[j])
    else:
        print("No results found")
        return

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot()

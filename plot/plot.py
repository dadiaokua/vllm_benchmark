import json
import math
import os
import matplotlib.pyplot as plt


def plot_result(args_concurrency, args_num_requests, total_time):
    with open('benchmark_results.json', 'r') as f:
        all_results = json.load(f)

    if len(all_results) >= 1:
        # Group results by short/long and sort by client_index
        short_results = []
        long_results = []

        for result in all_results:
            if "short" in result[0]["client_index"]:
                short_results.append(result)
            else:
                long_results.append(result)

        # Sort results by client_index
        short_results.sort(key=lambda x: x[0]["client_index"])
        long_results.sort(key=lambda x: x[0]["client_index"])

        # Combine sorted results with short first, then long
        sorted_all_results = short_results + long_results

        # Plot individual client results
        cols = math.ceil(math.sqrt(len(sorted_all_results)))
        rows = math.ceil(len(sorted_all_results) / cols)

        # 调整第一个图的大小，减小右侧留白
        fig1, axs1 = plt.subplots(rows * 3, cols, figsize=(cols * 8, rows * 12))
        # 添加大标题显示参数信息
        fig1.suptitle(
            f"Benchmark Results - Concurrency: {args_concurrency}, Requests Number: {args_num_requests}, Total Time: {total_time}",
            fontsize=16, y=0.99)

        # Handle different axes array shapes
        if rows == 1 and cols == 1:
            axs1 = axs1.reshape(3, 1)
        else:
            axs1 = axs1.reshape(rows * 3, cols)

        # 调整第二个图的大小
        fig2, axs2 = plt.subplots(3, 2, figsize=(20, 15))
        # 添加大标题显示参数信息
        fig2.suptitle(
            f"Averaged Benchmark Results - Concurrency: {args_concurrency}, Requests Number: {args_num_requests}, "
            f"Total Time: {total_time}", fontsize=16, y=0.99)

        # Plot individual client results
        for index, all_result in enumerate(sorted_all_results):
            request_number = []
            total_time = []
            total_output_tokens = []
            latency = []
            tokens_per_second = []
            time_to_first_token = []
            success_rate = []
            requests_per_second = []
            concurrency = []
            qps = []
            for i in range(len(all_result)):
                success_rate.append(all_result[i]['successful_requests'] / all_result[i]['total_requests'])
                request_number.append(all_result[i]["total_requests"])
                total_time.append(all_result[i]["total_time"])
                total_output_tokens.append(all_result[i]["total_output_tokens"])
                latency.append(all_result[i]["latency"]["p99"])
                tokens_per_second.append(all_result[i]["tokens_per_second"]["p99"])
                time_to_first_token.append(all_result[i]["time_to_first_token"]["p99"])
                requests_per_second.append(all_result[i]["requests_per_second"])
                concurrency.append(all_result[i]["concurrency"])
                qps.append(all_result[i]["qps"])

            row_idx = (index // cols) * 3
            col_idx = index % cols

            # First subplot: total_time
            axs1[row_idx][col_idx].plot(qps, total_time, label='total_time', color='blue', marker='o')
            axs1[row_idx][col_idx].set_title(f"Client {all_result[0]['client_index']} - Total Time: {sum(total_time):.2f}", pad=15)
            axs1[row_idx][col_idx].set_xlabel('qps')
            axs1[row_idx][col_idx].legend()
            axs1[row_idx][col_idx].grid(True, linestyle='--', alpha=0.7)

            # Second subplot: success_rate, tokens_per_second
            axs1[row_idx + 1][col_idx].plot(qps, success_rate, label='success_rate', color='pink',
                                            marker='s')
            axs1[row_idx + 1][col_idx].plot(qps, tokens_per_second, label='tokens_per_second', color='red',
                                            marker='x')
            for i, (x, y) in enumerate(zip(qps, success_rate)):
                axs1[row_idx + 1][col_idx].annotate(f'{y:.1f}',
                                                    (x, y),
                                                    textcoords="offset points",
                                                    xytext=(0, 10),
                                                    ha='center')
            for i, (x, y) in enumerate(zip(qps, tokens_per_second)):
                axs1[row_idx + 1][col_idx].annotate(f'{y:.1f}',
                                                    (x, y),
                                                    textcoords="offset points",
                                                    xytext=(0, -15),
                                                    ha='center')
            axs1[row_idx + 1][col_idx].set_title(f"Client {all_result[0]['client_index']} - Success Rate & Tokens/s",
                                                 pad=15)
            axs1[row_idx + 1][col_idx].set_xlabel('qps')
            axs1[row_idx + 1][col_idx].legend()
            axs1[row_idx + 1][col_idx].grid(True, linestyle='--', alpha=0.7)

            # Third subplot: latency, time_to_first_token, requests_per_second
            axs1[row_idx + 2][col_idx].plot(qps, latency, label='latency', color='green', marker='^')
            axs1[row_idx + 2][col_idx].plot(qps, time_to_first_token, label='time_to_first_token',
                                            color='purple', marker='*')
            axs1[row_idx + 2][col_idx].plot(qps, requests_per_second, label='requests_per_second',
                                            color='black', marker='1')
            axs1[row_idx + 2][col_idx].set_title(f"Client {all_result[0]['client_index']} - Latency Metrics", pad=15)
            axs1[row_idx + 2][col_idx].set_xlabel('qps')
            axs1[row_idx + 2][col_idx].legend()
            axs1[row_idx + 2][col_idx].grid(True, linestyle='--', alpha=0.7)

        # Remove extra subplots if any
        for j in range(len(sorted_all_results), cols):
            if rows * 3 > 1:
                for i in range(3):
                    fig1.delaxes(axs1[-3 + i][j])
            else:
                for i in range(3):
                    fig1.delaxes(axs1[i][j])

        # Plot averaged results for short/long
        for idx, (results, title) in enumerate([
            (short_results, "Short Context Average"),
            (long_results, "Long Context Average")
        ]):
            if not results:
                continue

            avg_total_time = []
            avg_success_rate = []
            avg_tokens_per_second = []
            avg_latency = []
            avg_ttft = []
            avg_rps = []
            concurrency = []
            qps = []

            # Get the shortest length among all results
            min_len = min(len(result) for result in results)

            for i in range(min_len):
                qps.append(results[0][i]["qps"])
                concurrency.append(results[0][i]["concurrency"])
                total_time = [result[i]["total_time"] for result in results]
                success_rate = [result[i]['successful_requests'] / result[i]['total_requests'] for result in
                                results]
                tokens_per_second = [result[i]["tokens_per_second"]["p99"] for result in results]
                latency = [result[i]["latency"]["p99"] for result in results]
                ttft = [result[i]["time_to_first_token"]["p99"] for result in results]
                rps = [result[i]["requests_per_second"] for result in results]

                avg_total_time.append(sum(total_time) / len(total_time))
                avg_success_rate.append(sum(success_rate) / len(success_rate))
                avg_tokens_per_second.append(sum(tokens_per_second) / len(tokens_per_second))
                avg_latency.append(sum(latency) / len(latency))
                avg_ttft.append(sum(ttft) / len(ttft))
                avg_rps.append(sum(rps) / len(rps))

            # Plot averaged metrics with enhanced styling
            axs2[0, idx].plot(qps, avg_total_time, label='total_time', color='blue', marker='o', linewidth=2)
            for i, (x, y) in enumerate(zip(qps, avg_total_time)):
                axs2[0, idx].annotate(f'{y:.1f}',
                                      (x, y),
                                      textcoords="offset points",
                                      xytext=(0, 10),
                                      ha='center')
            axs2[0, idx].set_title(f"{title} - Total Time: {sum(avg_total_time):.2f}", pad=15)
            axs2[0, idx].set_xlabel('qps')
            axs2[0, idx].legend()
            axs2[0, idx].grid(True, linestyle='--', alpha=0.7)

            axs2[1, idx].plot(qps, avg_success_rate, label='success_rate', color='pink', marker='s',
                              linewidth=2)
            axs2[1, idx].plot(qps, avg_tokens_per_second, label='tokens_per_second', color='red', marker='x',
                              linewidth=2)
            for i, (x, y) in enumerate(zip(qps, avg_success_rate)):
                axs2[1, idx].annotate(f'{y:.1f}',
                                      (x, y),
                                      textcoords="offset points",
                                      xytext=(0, 10),
                                      ha='center')
            for i, (x, y) in enumerate(zip(qps, avg_tokens_per_second)):
                axs2[1, idx].annotate(f'{y:.1f}',
                                      (x, y),
                                      textcoords="offset points",
                                      xytext=(0, -15),
                                      ha='center')
            axs2[1, idx].set_title(f"{title} - Success Rate & Tokens/s", pad=15)
            axs2[1, idx].set_xlabel('qps')
            axs2[1, idx].legend()
            axs2[1, idx].grid(True, linestyle='--', alpha=0.7)

            axs2[2, idx].plot(qps, avg_latency, label='latency', color='green', marker='^', linewidth=2)
            axs2[2, idx].plot(qps, avg_ttft, label='time_to_first_token', color='purple', marker='*',
                              linewidth=2)
            axs2[2, idx].plot(qps, avg_rps, label='requests_per_second', color='black', marker='1', linewidth=2)
            for i, (x, y) in enumerate(zip(qps, avg_latency)):
                axs2[2, idx].annotate(f'{y:.1f}',
                                      (x, y),
                                      textcoords="offset points",
                                      xytext=(0, 10),
                                      ha='center')
            for i, (x, y) in enumerate(zip(qps, avg_ttft)):
                axs2[2, idx].annotate(f'{y:.1f}',
                                      (x, y),
                                      textcoords="offset points",
                                      xytext=(0, -15),
                                      ha='center')
            for i, (x, y) in enumerate(zip(qps, avg_rps)):
                axs2[2, idx].annotate(f'{y:.1f}',
                                      (x, y),
                                      textcoords="offset points",
                                      xytext=(15, 0),
                                      ha='left')
            axs2[2, idx].set_title(f"{title} - Latency Metrics", pad=15)
            axs2[2, idx].set_xlabel('qps')
            axs2[2, idx].legend()
            axs2[2, idx].grid(True, linestyle='--', alpha=0.7)

    else:
        print("No results found")
        return

    # 优化布局
    fig1.tight_layout(pad=3.0, h_pad=2.0, w_pad=2.0)
    fig2.tight_layout(pad=3.0, h_pad=2.0, w_pad=2.0)

    # 调整大标题位置，避免与子图重叠
    fig1.subplots_adjust(top=0.95)
    fig2.subplots_adjust(top=0.95)

    # 创建 figure 文件夹（如果不存在）
    if not os.path.exists('../figure'):
        os.makedirs('../figure')

    # 保存图片
    fig1.savefig('figure/individual_clients.png', dpi=300, bbox_inches='tight')
    fig2.savefig('figure/averaged_results.png', dpi=300, bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    plot_result(25, 2000, 0)

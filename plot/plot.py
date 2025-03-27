import json
import math
import os
import matplotlib.pyplot as plt


def calculate_time_and_qps(time, qps):
    """Calculate adjusted time and format QPS with time."""
    for i in reversed(range(len(time))):
        if i == 0:
            time[i] = int(0)
            break
        time[i] = int(time[i] - time[0])
    return [f'{time[i]} [{qps[i]}]' for i in range(len(qps))]

def plot_metrics_with_annotations(ax, x_data, y_data, label, color, marker, linewidth=2, xytext=(0,10), linestyle='-'):
    """Plot metrics with annotations."""
    ax.plot(x_data, y_data, label=label, color=color, marker=marker, linewidth=linewidth, linestyle=linestyle)
    for i, (x, y) in enumerate(zip(x_data, y_data)):
        ax.annotate(f'{y:.1f}',
                   (x, y),
                   textcoords="offset points", 
                   xytext=xytext,
                   ha='center')

def setup_subplot(ax, title, qps_with_time, y_format=None, ylim=None):
    """Setup common subplot properties."""
    ax.set_title(title, pad=15)
    ax.set_xticks(range(len(qps_with_time)))
    ax.set_xticklabels(qps_with_time)
    ax.set_xlabel('time [qps]')
    if y_format:
        ax.yaxis.set_major_formatter(y_format)
    if ylim:
        ax.set_ylim(ylim)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

def plot_result(filename, args_concurrency, args_num_requests, total_time):
    with open("results/" + filename, 'r') as f:
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
        fig1, axs1 = plt.subplots(rows * 4, cols, figsize=(cols * 8, rows * 16))  # Changed from rows * 3 to rows * 4
        # 添加大标题显示参数信息
        fig1.suptitle(
            f"Benchmark Results - Concurrency: {args_concurrency}, Total Time: {total_time}",
            fontsize=16, y=0.99)

        # Handle different axes array shapes
        if rows == 1 and cols == 1:
            axs1 = axs1.reshape(4, 1)  # Changed from 3 to 4
        else:
            axs1 = axs1.reshape(rows * 4, cols)  # Changed from rows * 3 to rows * 4

        # 调整第二个图的大小
        fig2, axs2 = plt.subplots(4, 2, figsize=(20, 20))  # Changed from 3, 2 to 4, 2
        # 添加大标题显示参数信息
        fig2.suptitle(
            f"Averaged Benchmark Results - Concurrency: {args_concurrency}, "
            f"Total Time: {total_time}", fontsize=16, y=0.99)

        # 定义不同的线条样式
        line_styles = ['-', '--', '-.', ':']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        markers = ['o', 's', '^', 'D', 'v', '*']

        # Plot individual client results
        for index, all_result in enumerate(sorted_all_results):
            request_number = []
            success_rate = []
            total_output_tokens = []
            total_input_tokens = []
            latency = []
            tokens_per_second = []
            time_to_first_token = []
            requests_per_second = []
            concurrency = []
            qps = []
            time = []
            tokens_count = []
            for i in range(len(all_result)):
                tokens_count.append(all_result[i]['total_output_tokens'] + all_result[i]['total_input_tokens'])
                total_output_tokens.append(all_result[i]["total_output_tokens"])
                total_input_tokens.append(all_result[i]["total_input_tokens"])
                time.append(all_result[i]['time'])
                success_rate.append(all_result[i]['successful_requests'] * 100 / all_result[i]['total_requests'])
                request_number.append(all_result[i]["total_requests"])
                latency.append(all_result[i]["latency"]["p99"])
                tokens_per_second.append(all_result[i]["tokens_per_second"]["p99"])
                time_to_first_token.append(all_result[i]["time_to_first_token"]["p99"])
                requests_per_second.append(all_result[i]["requests_per_second"])
                concurrency.append(all_result[i]["concurrency"])
                qps.append(all_result[i]["qps"])

            qps_with_time = calculate_time_and_qps(time, qps)

            row_idx = (index // cols) * 4
            col_idx = index % cols

            # First subplot: tokens count
            plot_metrics_with_annotations(axs1[row_idx][col_idx], range(len(qps)), tokens_count, 'tokens count', 
                                       colors[0], markers[0], linestyle=line_styles[0])
            plot_metrics_with_annotations(axs1[row_idx][col_idx], range(len(qps)), total_output_tokens, 'total_output_tokens',
                                       colors[1], markers[1], linestyle=line_styles[1])
            plot_metrics_with_annotations(axs1[row_idx][col_idx], range(len(qps)), total_input_tokens, 'total_input_tokens',
                                       colors[2], markers[2], linestyle=line_styles[2])
            setup_subplot(axs1[row_idx][col_idx], 
                         f"Client {all_result[0]['client_index']} - Input and Output Tokens",
                         qps_with_time,
                         plt.FuncFormatter(lambda x, p: format(int(x), ',')))

            # Second subplot: success_rate
            plot_metrics_with_annotations(axs1[row_idx + 1][col_idx], range(len(qps)), success_rate, 'success_rate',
                                       colors[3], markers[3], linestyle=line_styles[0])
            setup_subplot(axs1[row_idx + 1][col_idx],
                         f"Client {all_result[0]['client_index']} - Success Rate",
                         qps_with_time,
                         ylim=(max(min(success_rate)-10,0), 105))

            # Third subplot: tokens per second
            plot_metrics_with_annotations(axs1[row_idx + 2][col_idx], range(len(qps)), tokens_per_second, 'tokens_per_second',
                                       colors[4], markers[4], xytext=(0,-15), linestyle=line_styles[0])
            setup_subplot(axs1[row_idx + 2][col_idx],
                         f"Client {all_result[0]['client_index']} - Tokens/s",
                         qps_with_time)

            # Fourth subplot: latency, time_to_first_token, requests_per_second
            plot_metrics_with_annotations(axs1[row_idx + 3][col_idx], range(len(qps)), latency, 'latency',
                                       colors[0], markers[0], linestyle=line_styles[0])
            plot_metrics_with_annotations(axs1[row_idx + 3][col_idx], range(len(qps)), time_to_first_token, 'time_to_first_token',
                                       colors[1], markers[1], linestyle=line_styles[1])
            plot_metrics_with_annotations(axs1[row_idx + 3][col_idx], range(len(qps)), requests_per_second, 'requests_per_second',
                                       colors[2], markers[2], linestyle=line_styles[2])
            setup_subplot(axs1[row_idx + 3][col_idx],
                         f"Client {all_result[0]['client_index']} - Latency Metrics",
                         qps_with_time)

        # Remove extra subplots if any
        for j in range(len(sorted_all_results), cols):
            if rows * 4 > 1:
                for i in range(4):
                    fig1.delaxes(axs1[-4 + i][j])
            else:
                for i in range(4):
                    fig1.delaxes(axs1[i][j])

        # Plot averaged results for short/long
        for idx, (results, title) in enumerate([
            (short_results, "Short Context Average"),
            (long_results, "Long Context Average")
        ]):
            if not results:
                continue

            avg_success_rate = []
            avg_tokens_per_second = []
            avg_latency = []
            avg_ttft = []
            avg_rps = []
            avg_tokens_count = []
            avg_total_output_tokens = []
            avg_total_input_tokens = []
            concurrency = []
            qps = []
            time = []

            # Get the shortest length among all results
            min_len = min(len(result) for result in results)

            for i in range(min_len):
                time.append(results[0][i]["time"])
                # Multiply QPS by number of clients based on request type
                if title == "Short Context Average":
                    qps.append(results[0][i]["qps"] * len(short_results))
                else:
                    qps.append(results[0][i]["qps"] * len(long_results))
                concurrency.append(results[0][i]["concurrency"])

                tokens_count = [result[i]['total_output_tokens'] + result[i]['total_input_tokens'] for result in results]
                total_output_tokens = [result[i]['total_output_tokens'] for result in results]
                total_input_tokens = [result[i]['total_input_tokens'] for result in results]
                success_rate = [result[i]['successful_requests'] * 100 / result[i]['total_requests'] for result in results]
                tokens_per_second = [result[i]["tokens_per_second"]["p99"] for result in results]
                latency = [result[i]["latency"]["p99"] for result in results]
                ttft = [result[i]["time_to_first_token"]["p99"] for result in results]
                rps = [result[i]["requests_per_second"] for result in results]

                avg_tokens_count.append(sum(tokens_count) / len(tokens_count))
                avg_total_output_tokens.append(sum(total_output_tokens) / len(total_output_tokens))
                avg_total_input_tokens.append(sum(total_input_tokens) / len(total_input_tokens))
                avg_success_rate.append(sum(success_rate) / len(success_rate))
                avg_tokens_per_second.append(sum(tokens_per_second) / len(tokens_per_second))
                avg_latency.append(sum(latency) / len(latency))
                avg_ttft.append(sum(ttft) / len(ttft))
                avg_rps.append(sum(rps) / len(rps))

            qps_with_time = calculate_time_and_qps(time, qps)

            # Plot tokens count with different line styles
            plot_metrics_with_annotations(axs2[0, idx], range(len(qps)), avg_tokens_count, 'tokens count',
                                       colors[0], markers[0], linestyle=line_styles[0])
            plot_metrics_with_annotations(axs2[0, idx], range(len(qps)), avg_total_output_tokens, 'total_output_tokens',
                                       colors[1], markers[1], linestyle=line_styles[1])
            plot_metrics_with_annotations(axs2[0, idx], range(len(qps)), avg_total_input_tokens, 'total_input_tokens',
                                       colors[2], markers[2], linestyle=line_styles[2])
            setup_subplot(axs2[0, idx],
                         f"{title} - Input and Output Tokens",
                         qps_with_time,
                         plt.FuncFormatter(lambda x, p: format(int(x), ',')))

            # Plot success rate
            plot_metrics_with_annotations(axs2[1, idx], range(len(qps)), avg_success_rate, 'success_rate',
                                       colors[3], markers[3], linestyle=line_styles[0])
            setup_subplot(axs2[1, idx],
                         f"{title} - Success Rate",
                         qps_with_time,
                         ylim=(max(min(avg_success_rate)-10,0), 105))

            # Plot tokens per second
            plot_metrics_with_annotations(axs2[2, idx], range(len(qps)), avg_tokens_per_second, 'tokens_per_second',
                                       colors[4], markers[4], xytext=(0,-15), linestyle=line_styles[0])
            setup_subplot(axs2[2, idx],
                         f"{title} - Tokens/s",
                         qps_with_time)

            # Plot latency metrics
            plot_metrics_with_annotations(axs2[3, idx], range(len(qps)), avg_latency, 'latency',
                                       colors[0], markers[0], linestyle=line_styles[0])
            plot_metrics_with_annotations(axs2[3, idx], range(len(qps)), avg_ttft, 'time_to_first_token',
                                       colors[1], markers[1], xytext=(0,-15), linestyle=line_styles[1])
            plot_metrics_with_annotations(axs2[3, idx], range(len(qps)), avg_rps, 'requests_per_second',
                                       colors[2], markers[2], xytext=(15,0), linestyle=line_styles[2])
            setup_subplot(axs2[3, idx],
                         f"{title} - Latency Metrics",
                         qps_with_time)

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
    fig1.savefig('figure/individual_clients' + filename + '.png', dpi=300, bbox_inches='tight')
    fig2.savefig('figure/averaged_results' + filename + '.png', dpi=300, bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    with open("plot_data.json", "r") as f:
        data = json.load(f)

    plot_result(data["filename"], data["concurrency"], data["num_requests"], data["total_time"])

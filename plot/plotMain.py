import json
import math
import os
import matplotlib.pyplot as plt

from plot.plotUtil import xLabel_time, plot_metrics_with_annotations, setup_subplot, setup_subplot_client

line_styles = ['-', '--', '-.', ':']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
markers = ['o', 's', '^', 'D', 'v', '*']


def plot_averaged_results(short_results, long_results, args_concurrency, total_time, filename, exp_type):
    # 创建一个4×2的子图布局,宽高比更大
    fig2, axs2 = plt.subplots(4, 2, figsize=(28, 20))

    # 添加标题显示参数
    fig2.suptitle(
        f"{exp_type} Averaged Benchmark Results - Concurrency: {args_concurrency}, Total Time: {total_time}",
        fontsize=16, y=0.98)

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

        time_xLabel = xLabel_time(time)

        # Plot tokens count with different line styles
        plot_metrics_with_annotations(axs2[0, idx], range(len(qps)), avg_tokens_count, 'tokens count',
                                      colors[0], markers[0], linestyle=line_styles[0])
        plot_metrics_with_annotations(axs2[0, idx], range(len(qps)), avg_total_output_tokens, 'total_output_tokens',
                                      colors[1], markers[1], linestyle=line_styles[1])
        plot_metrics_with_annotations(axs2[0, idx], range(len(qps)), avg_total_input_tokens, 'total_input_tokens',
                                      colors[2], markers[2], linestyle=line_styles[2])
        setup_subplot(axs2[0, idx],
                      f"{title} - Input and Output Tokens",
                      time_xLabel,
                      plt.FuncFormatter(lambda x, p: format(int(x), ',')))

        # Plot success rate
        plot_metrics_with_annotations(axs2[1, idx], range(len(qps)), avg_success_rate, 'success_rate',
                                      colors[3], markers[3], linestyle=line_styles[0])
        setup_subplot(axs2[1, idx],
                      f"{title} - Success Rate",
                      time_xLabel,
                      ylim=(max(min(avg_success_rate) - 10, 0), 105))

        # Plot tokens per second
        plot_metrics_with_annotations(axs2[2, idx], range(len(qps)), avg_tokens_per_second, 'tokens_per_second',
                                      colors[4], markers[4], xytext=(0, -15), linestyle=line_styles[0])
        setup_subplot(axs2[2, idx],
                      f"{title} - Tokens/s",
                      time_xLabel)

        # Plot latency metrics
        plot_metrics_with_annotations(axs2[3, idx], range(len(qps)), avg_latency, 'latency',
                                      colors[0], markers[0], linestyle=line_styles[0])
        plot_metrics_with_annotations(axs2[3, idx], range(len(qps)), avg_ttft, 'time_to_first_token',
                                      colors[1], markers[1], xytext=(0, -15), linestyle=line_styles[1])
        plot_metrics_with_annotations(axs2[3, idx], range(len(qps)), avg_rps, 'requests_per_second',
                                      colors[2], markers[2], xytext=(15, 0), linestyle=line_styles[2])
        setup_subplot(axs2[3, idx],
                      f"{title} - Latency Metrics",
                      time_xLabel)

    # 优化布局
    fig2.tight_layout(pad=3.0, h_pad=2.0, w_pad=2.0)
    # 调整子图间距，为底部图例留出空间
    fig2.subplots_adjust(top=0.92, bottom=0.1)

    # 创建 figure 文件夹（如果不存在）
    if not os.path.exists('../figure'):
        os.makedirs('../figure')

    # 保存图片
    fig2.savefig('figure/averaged_results' + filename + '.png', dpi=300, bbox_inches='tight')
    plt.show()
    return time_xLabel


def plot_comprehensive_results(sorted_all_results, args_concurrency, total_time, filename, exp_type, fairness_results,
                               qps_with_time):
    """
    绘制综合性能图表，包括各客户端指标和公平性结果
    """
    # 创建一个4×2的子图布局,宽高比更大
    fig, axs = plt.subplots(4, 2, figsize=(28, 20))
    axs = axs.flatten()  # 将2D数组展平为1D，便于索引

    # 添加标题显示参数
    fig.suptitle(
        f"{exp_type} Benchmark Results - Concurrency: {args_concurrency}, Total Time: {total_time}",
        fontsize=16, y=0.98)

    # 获取所有客户端
    clients = set()
    for result in sorted_all_results:
        clients.add(result[0]["client_index"])

    # 将clients分为short和long两组
    short_clients = sorted([c for c in clients if "short" in c])
    long_clients = sorted([c for c in clients if "long" in c])

    print(f"Short clients: {short_clients}")
    print(f"Long clients: {long_clients}")

    # 暖色系用于short clients
    warm_colors = ['#FF4D4D', '#FFA64D', '#FFD700', '#FF69B4', '#FF8C69']
    # 冷色系用于long clients  
    cool_colors = ['#4169E1', '#00CED1', '#6495ED', '#483D8B', '#008B8B']

    # 创建一个共享图例的句柄和标签列表
    legend_handles = []
    legend_labels = []

    # 绘制各个客户端的指标
    # 1. 成功率
    plot_client_metric(axs[0], sorted_all_results, short_clients, long_clients,
                       warm_colors, cool_colors, "success_rate",
                       "Success Rate (%)", legend_handles, legend_labels)

    # 2. 每秒令牌数
    plot_client_metric(axs[1], sorted_all_results, short_clients, long_clients,
                       warm_colors, cool_colors, "tokens_per_second",
                       "Tokens/s", legend_handles, legend_labels)

    # 3. 延迟
    plot_client_metric(axs[2], sorted_all_results, short_clients, long_clients,
                       warm_colors, cool_colors, "latency",
                       "Latency (ms)", legend_handles, legend_labels)

    # 4. 首个令牌时间
    plot_client_metric(axs[3], sorted_all_results, short_clients, long_clients,
                       warm_colors, cool_colors, "time_to_first_token",
                       "Time to First Token (ms)", legend_handles, legend_labels)

    # 5. 每秒请求数
    plot_client_metric(axs[4], sorted_all_results, short_clients, long_clients,
                       warm_colors, cool_colors, "requests_per_second",
                       "Requests per Second", legend_handles, legend_labels)

    times = [q for q in qps_with_time]

    # 6. 每个客户端的QPS
    plot_client_metric(axs[5], sorted_all_results, short_clients, long_clients,
                       warm_colors, cool_colors, "qps",
                       "QPS Per Client Over Time", legend_handles, legend_labels)

    # 7. 每个客户端的服务使用量
    plot_client_metric(axs[6], sorted_all_results, short_clients, long_clients,
                       warm_colors, cool_colors, "service",
                       "Service Usage", legend_handles, legend_labels)

    # 8. Jain's公平性指数放在最后
    plot_fairness_index(axs[7], fairness_results, times)

    # 在图形底部添加共享图例
    fig.legend(handles=legend_handles, labels=legend_labels,
               loc='center', bbox_to_anchor=(0.5, 0.02),
               ncol=len(short_clients) + len(long_clients),
               fontsize=10)

    # 优化布局
    fig.tight_layout(pad=3.0, h_pad=2.0, w_pad=2.0)
    # 调整子图间距，为底部图例留出空间
    fig.subplots_adjust(top=0.92, bottom=0.1)

    # 创建figure文件夹（如果不存在）
    if not os.path.exists('figure'):
        os.makedirs('figure')

    # 保存图表
    fig.savefig(f'figure/comprehensive_results{filename}.png', dpi=300, bbox_inches='tight')

    plt.show()


def plot_client_metric(ax, sorted_all_results, short_clients, long_clients, warm_colors, cool_colors, metric_key, title,
                       legend_handles, legend_labels, ylim=None):
    """绘制客户端指标"""
    # 先绘制short clients，使用实线
    for i, client in enumerate(short_clients):
        client_results = next(r for r in sorted_all_results if r[0]["client_index"] == client)
        time = [result['time'] for result in client_results]
        time_xLabel = xLabel_time(time)

        if metric_key == "success_rate":
            values = [result['successful_requests'] * 100 / result['total_requests'] for result in client_results]
        elif metric_key in ["tokens_per_second", "latency", "time_to_first_token"]:
            values = [result[metric_key]["p99"] for result in client_results]
        elif metric_key == "service":
            values = [result['total_input_tokens'] + result['total_output_tokens'] * 2 for result in client_results]
        else:
            values = [result[metric_key] for result in client_results]

        line = ax.plot(range(len(time)), values, marker=markers[i % len(markers)],
                       color=warm_colors[i % len(warm_colors)], linestyle='-',
                       label=client)[0]
        if client not in legend_labels:
            legend_handles.append(line)
            legend_labels.append(client)

    # 再绘制long clients，使用虚线
    for i, client in enumerate(long_clients):
        client_results = next(r for r in sorted_all_results if r[0]["client_index"] == client)
        time = [result['time'] for result in client_results]
        time_xLabel = xLabel_time(time)

        if metric_key == "success_rate":
            values = [result['successful_requests'] * 100 / result['total_requests'] for result in client_results]
        elif metric_key in ["tokens_per_second", "latency", "time_to_first_token"]:
            values = [result[metric_key]["p99"] for result in client_results]
        elif metric_key == "service":
            values = [result['total_input_tokens'] + result['total_output_tokens'] * 2 for result in client_results]
        else:
            values = [result[metric_key] for result in client_results]

        line = ax.plot(range(len(time)), values, marker=markers[i % len(markers)],
                       color=cool_colors[i % len(cool_colors)], linestyle='--',
                       label=client)[0]
        if client not in legend_labels:
            legend_handles.append(line)
            legend_labels.append(client)

    # 计算所有值以确定合适的y轴范围
    all_values = []
    for client in short_clients + long_clients:
        try:
            client_results = next(r for r in sorted_all_results if r[0]["client_index"] == client)
            if metric_key == "success_rate":
                values = [result['successful_requests'] * 100 / result['total_requests'] for result in client_results]
            elif metric_key in ["tokens_per_second", "latency", "time_to_first_token"]:
                values = [result[metric_key]["p99"] for result in client_results]
            elif metric_key == "service":
                values = [result['total_input_tokens'] + result['total_output_tokens'] * 2 for result in client_results]
            else:
                values = [result[metric_key] for result in client_results]
            all_values.extend(values)
        except (StopIteration, KeyError, ZeroDivisionError) as e:
            print(f"Warning: Error processing {client} for {metric_key}: {e}")

    # 如果没有提供ylim且有足够的值，则自动计算合适的范围
    if ylim is None and all_values:
        try:
            min_val = min(v for v in all_values if v is not None)
            max_val = max(v for v in all_values if v is not None)
            margin = (max_val - min_val) * 0.1
            computed_ylim = (max(0, min_val - margin), max_val + margin)
            if metric_key == "success_rate":
                computed_ylim = (max(0, min_val - 5), 105)
        except (ValueError, TypeError):
            computed_ylim = None
    else:
        computed_ylim = ylim

    # 设置子图属性
    setup_subplot_client(ax, title, time_xLabel, ylim=computed_ylim)


def plot_fairness_index(ax, fairness_results, times):
    """绘制Jain's公平性指数"""
    f_results = [result["f_result"] for result in fairness_results]
    line, = ax.plot(times, f_results, marker='o', color='#1f77b4', linestyle='-', linewidth=2)

    # 在线上显示数值
    for i, (x, y) in enumerate(zip(times, f_results)):
        ax.annotate(f'{y:.3f}',
                    xy=(x, y),
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

    ax.set_title("Jain's Fairness Index Over Time", fontsize=12, pad=10)
    ax.set_xlabel("Time(s)", fontsize=10)
    ax.set_ylabel("Fairness Index", fontsize=10)
    ax.tick_params(axis='both', labelsize=9)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_ylim(0, 1)


def plot_result(exp_type, filename, args_concurrency, total_time):
    with open("results/" + filename, 'r') as f:
        all_results = json.load(f)

    # Load fairness results and create third figure
    with open("tmp_result/tmp_fairness_result.json", 'r') as f:
        fairness_results = json.load(f)

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

        qps_with_time = plot_averaged_results(short_results, long_results, args_concurrency, total_time, filename,
                                              exp_type)
        plot_comprehensive_results(sorted_all_results, args_concurrency, total_time, filename, exp_type,
                                   fairness_results, qps_with_time)
    else:
        print("No results found")
        return


if __name__ == "__main__":
    with open("tmp_result/plot_data.json", "r") as f:
        data = json.load(f)

    plot_result(data["exp_type"], data["filename"], data["concurrency"], data["total_time"])

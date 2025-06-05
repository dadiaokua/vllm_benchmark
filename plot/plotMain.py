import json
import math
import os
import matplotlib.pyplot as plt
import numpy as np

from config.Config import GLOBAL_CONFIG
from plot.plotUtil import xLabel_time, plot_metrics_with_annotations, setup_subplot, setup_subplot_client, \
    plot_fairness_index

line_styles = ['-', '--', '-.', ':']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
markers = ['o', 's', '^', 'D', 'v', '*']


def plot_averaged_results(short_results, long_results, args_concurrency, total_time, filename, exp_type):
    # 创建一个4×1的子图布局
    fig2, axs2 = plt.subplots(4, 1, figsize=(14, 20))

    # 添加标题显示参数
    fig2.suptitle(
        f"{exp_type} System Benchmark Results - Short Client: {args_concurrency}, Total Time: {total_time}, Alpha: {GLOBAL_CONFIG.get('alpha', 0.5)}",
        fontsize=16, y=0.98)

    # 合并所有结果
    all_results = short_results + long_results

    if not all_results:
        return

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
    min_len = min(len(result) for result in all_results)

    for i in range(min_len):
        time.append(all_results[0][i]["time"])
        # 总QPS是所有客户端QPS之和
        qps.append(all_results[0][i]["qps"] * len(all_results))
        concurrency.append(all_results[0][i]["concurrency"])

        # 计算总和而不是平均值
        tokens_count = sum(result[i]['total_output_tokens'] + result[i]['total_input_tokens'] for result in all_results)
        total_output_tokens = sum(result[i]['total_output_tokens'] for result in all_results)
        total_input_tokens = sum(result[i]['total_input_tokens'] for result in all_results)
        # 成功请求数总和
        successful_requests = sum(result[i]['successful_requests'] for result in all_results)
        total_requests = sum(result[i]['total_requests'] for result in all_results)
        success_rate = successful_requests * 100 / total_requests if total_requests > 0 else 0
        # 取所有客户端延迟的和作为系统延迟
        tokens_per_second = sum(result[i]["tokens_per_second"]["p99"] for result in all_results)
        latency = max(result[i]["latency"]["p99"] for result in all_results)
        ttft = max(result[i]["time_to_first_token"]["p99"] for result in all_results)
        # 总RPS是所有客户端RPS之和
        rps = sum(result[i]["requests_per_second"] for result in all_results)

        avg_tokens_count.append(tokens_count)
        avg_total_output_tokens.append(total_output_tokens)
        avg_total_input_tokens.append(total_input_tokens)
        avg_success_rate.append(success_rate)
        avg_tokens_per_second.append(tokens_per_second)
        avg_latency.append(latency)
        avg_ttft.append(ttft)
        avg_rps.append(rps)

    time_xLabel = xLabel_time(time)

    # Plot tokens count with different line styles
    plot_metrics_with_annotations(axs2[0], range(len(qps)), avg_tokens_count, 'tokens count',
                                  colors[0], markers[0], linestyle=line_styles[0])
    plot_metrics_with_annotations(axs2[0], range(len(qps)), avg_total_output_tokens, 'total_output_tokens',
                                  colors[1], markers[1], linestyle=line_styles[1])
    plot_metrics_with_annotations(axs2[0], range(len(qps)), avg_total_input_tokens, 'total_input_tokens',
                                  colors[2], markers[2], linestyle=line_styles[2])
    setup_subplot(axs2[0],
                  "Sum Input and Output Tokens",
                  time_xLabel,
                  plt.FuncFormatter(lambda x, p: format(int(x), ',')))

    # Plot success rate
    plot_metrics_with_annotations(axs2[1], range(len(qps)), avg_success_rate, 'success_rate',
                                  colors[3], markers[3], linestyle=line_styles[0])
    setup_subplot(axs2[1],
                  "Average Success Rate",
                  time_xLabel,
                  ylim=(max(min(avg_success_rate) - 10, 0), 105))

    # Plot tokens per second
    plot_metrics_with_annotations(axs2[2], range(len(qps)), avg_tokens_per_second, 'tokens_per_second',
                                  colors[4], markers[4], xytext=(0, -15), linestyle=line_styles[0])
    setup_subplot(axs2[2],
                  "Sum Tokens/s",
                  time_xLabel)

    # Plot latency metrics
    plot_metrics_with_annotations(axs2[3], range(len(qps)), avg_latency, 'latency',
                                  colors[0], markers[0], linestyle=line_styles[0])
    plot_metrics_with_annotations(axs2[3], range(len(qps)), avg_ttft, 'time_to_first_token',
                                  colors[1], markers[1], xytext=(0, -15), linestyle=line_styles[1])
    plot_metrics_with_annotations(axs2[3], range(len(qps)), avg_rps, 'requests_per_second',
                                  colors[2], markers[2], xytext=(15, 0), linestyle=line_styles[2])
    setup_subplot(axs2[3],
                  "Max Latency and TTFT, Sum RPS Metrics",
                  time_xLabel)

    # 优化布局
    fig2.tight_layout(pad=3.0, h_pad=2.0)
    # 调整子图间距，为底部图例留出空间
    fig2.subplots_adjust(top=0.92, bottom=0.1)

    # 创建 figure 文件夹（如果不存在）
    if not os.path.exists('figure'):
        os.makedirs('figure')

    # 保存图片
    fig2.savefig('figure/system_results' + filename.split('.')[0] + '.png', dpi=300, bbox_inches='tight')
    plt.show()
    return time_xLabel


def plot_comprehensive_results(sorted_all_results, args_concurrency, total_time, filename, exp_type, fairness_results,
                               qps_with_time):
    """
    绘制综合性能图表，包括各客户端指标和公平性结果
    """
    # 创建两个图：性能指标图和公平性指标图
    fig1, axs1 = plt.subplots(4, 2, figsize=(28, 25))  # 性能指标图
    fig2, axs2 = plt.subplots(3, 1, figsize=(28, 18))   # 公平性指标图，改为2行1列
    axs1 = axs1.flatten()  # 将2D数组展平为1D，便于索引
    axs2 = axs2.flatten()

    # 添加标题显示参数
    fig1.suptitle(
        f"{exp_type} Performance Metrics - Concurrency: {args_concurrency}, Total Time: {total_time}, Alpha: {GLOBAL_CONFIG.get('alpha', 0.5)}",
        fontsize=16, y=0.98)
    fig2.suptitle(
        f"{exp_type} Fairness Metrics - Concurrency: {args_concurrency}, Total Time: {total_time}, Alpha: {GLOBAL_CONFIG.get('alpha', 0.5)}",
        fontsize=16, y=0.98)

    # 获取所有客户端
    clients = set()
    for result in sorted_all_results:
        clients.add(f'{result[0]["client_index"]}_slo_{result[0]["latency_slo"]}')

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

    # 绘制性能指标
    # 1. 成功率
    plot_client_metric(axs1[0], sorted_all_results, short_clients, long_clients,
                       warm_colors, cool_colors, "success_rate",
                       "Success Rate (%)", legend_handles, legend_labels)

    # 2. 每秒令牌数
    plot_client_metric(axs1[1], sorted_all_results, short_clients, long_clients,
                       warm_colors, cool_colors, "tokens_per_second",
                       "Tokens/s", legend_handles, legend_labels)

    # 3. 延迟
    plot_client_metric(axs1[2], sorted_all_results, short_clients, long_clients,
                       warm_colors, cool_colors, "latency",
                       "Latency (ms)", legend_handles, legend_labels)

    # 4. 首个令牌时间
    plot_client_metric(axs1[3], sorted_all_results, short_clients, long_clients,
                       warm_colors, cool_colors, "time_to_first_token",
                       "Time to First Token (ms)", legend_handles, legend_labels)

    # 5. 每秒请求数
    plot_client_metric(axs1[4], sorted_all_results, short_clients, long_clients,
                       warm_colors, cool_colors, "requests_per_second",
                       "Requests per Second", legend_handles, legend_labels)

    # 6. 每个客户端的服务使用量
    plot_client_metric(axs1[5], sorted_all_results, short_clients, long_clients,
                       warm_colors, cool_colors, "service",
                       "Service Usage", legend_handles, legend_labels)

    # 7. 成功请求数
    plot_client_metric(axs1[6], sorted_all_results, short_clients, long_clients,
                       warm_colors, cool_colors, "successful_requests",
                       "Successful Requests", legend_handles, legend_labels)

    # 8. 违反SLO的请求数量
    plot_client_metric(axs1[7], sorted_all_results, short_clients, long_clients,
                       warm_colors, cool_colors, "slo_violation_count",
                       "Slo Violation Request Numbers", legend_handles, legend_labels)

    # 绘制公平性指标
    # 1. Fairness Ratio
    plot_client_metric(axs2[0], sorted_all_results, short_clients, long_clients,
                       warm_colors, cool_colors, "fairness_ratio",
                       "Fairness Ratio", legend_handles, legend_labels)

    # 2. Jain's公平性指数
    f_values = [result['f_result'] for result in fairness_results]
    times = list(range(len(f_values)))
    if len(times) == len(f_values):
        plot_fairness_index(axs2[1], f_values, times)
    else:
        print(f"Warning: Mismatched lengths - times: {len(times)}, f_values: {len(f_values)}")
        min_len = min(len(times), len(f_values))
        plot_fairness_index(axs2[1], f_values[:min_len], times[:min_len])
    
    # 3. credit值
    plot_client_metric(axs2[2], sorted_all_results, short_clients, long_clients,
                       warm_colors, cool_colors, "credit",
                       "Clients Credit", legend_handles, legend_labels, ylim=None)

    # 为第三张图添加图例
    axs2[2].legend()

    # 在性能指标图底部添加共享图例
    fig1.legend(handles=legend_handles, labels=legend_labels,
               loc='center', bbox_to_anchor=(0.5, 0.02),
               ncol=len(short_clients) + len(long_clients),
               fontsize=10)

    # 优化布局
    fig1.tight_layout(pad=3.0, h_pad=2.0, w_pad=2.0)
    fig2.tight_layout(pad=3.0, h_pad=2.0, w_pad=2.0)
    # 调整子图间距，为底部图例留出空间
    fig1.subplots_adjust(top=0.92, bottom=0.1)
    fig2.subplots_adjust(top=0.92, bottom=0.1)

    # 创建figure文件夹（如果不存在）
    if not os.path.exists('figure'):
        os.makedirs('figure')

    # 保存图表
    fig1.savefig(f'figure/performance_metrics{filename.split(".")[0]}.png', dpi=300, bbox_inches='tight')
    fig2.savefig(f'figure/fairness_metrics{filename.split(".")[0]}.png', dpi=300, bbox_inches='tight')

    plt.show()


def plot_client_metric(ax, sorted_all_results, short_clients, long_clients, warm_colors, cool_colors, metric_key, title,
                       legend_handles, legend_labels, ylim=None):
    plotted_value_sets = []
    """绘制客户端指标"""
    # 先绘制short clients，使用实线
    for i, client in enumerate(short_clients):
        client_results = next(
            r for r in sorted_all_results if f'{r[0]["client_index"]}_slo_{r[0]["latency_slo"]}' == client)
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

        is_duplicate = any(np.allclose(values, prev, atol=1e-4) for prev in plotted_value_sets)
        if is_duplicate:
            x_vals = [j + 0.02 for j in range(len(time))]  # 偏移一点
        else:
            x_vals = list(range(len(time)))

        # 使用不同的marker和linestyle来区分不同的线条
        line = ax.plot(x_vals, values, marker=markers[i % len(markers)],
                       color=warm_colors[i % len(warm_colors)], linestyle='-',
                       label=client, alpha=0.7, markersize=8)[0]

        # 👇 保存已画过的 values
        plotted_value_sets.append(values)

        if client not in legend_labels:
            legend_handles.append(line)
            legend_labels.append(client)

    # 再绘制long clients，使用虚线
    for i, client in enumerate(long_clients):
        client_results = next(
            r for r in sorted_all_results if f'{r[0]["client_index"]}_slo_{r[0]["latency_slo"]}' == client)
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

        is_duplicate = any(np.allclose(values, prev, atol=1e-4) for prev in plotted_value_sets)
        if is_duplicate:
            x_vals = [j + 0.02 for j in range(len(time))]  # 偏移一点
        else:
            x_vals = list(range(len(time)))

        # 使用不同的marker和linestyle来区分不同的线条
        line = ax.plot(x_vals, values, marker=markers[(i + len(short_clients)) % len(markers)],
                       color=cool_colors[i % len(cool_colors)], linestyle='--',
                       label=client, alpha=0.7, markersize=8)[0]

        plotted_value_sets.append(values)

        if client not in legend_labels:
            legend_handles.append(line)
            legend_labels.append(client)

    # 计算所有值以确定合适的y轴范围
    all_values = []
    for client in short_clients + long_clients:
        try:
            client_results = next(
                r for r in sorted_all_results if f'{r[0]["client_index"]}_slo_{r[0]["latency_slo"]}' == client)
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

    print(f"[DEBUG] {metric_key} all_values: {all_values}")

    # 如果没有提供ylim且有足够的值，则自动计算合适的范围
    if ylim is None and all_values:
        try:
            # 过滤掉None值
            valid_values = [v for v in all_values if v is not None]
            if valid_values:  # 确保有有效值
                min_val = min(valid_values)
                max_val = max(valid_values)
                # 增加最小margin，避免y轴范围过小
                margin = max((max_val - min_val) * 0.1, 1.0)
                if min_val < 0:
                    computed_ylim = (min_val - margin, max_val + margin)
                else:
                    computed_ylim = (max(0, min_val - margin), max_val + margin)
                # 特殊处理 success_rate
                if metric_key == "success_rate":
                    computed_ylim = (max(0, min_val - 5), 105)
            else:
                computed_ylim = None  # 如果没有有效值，设置为None
        except (ValueError, TypeError) as e:
            print(f"Warning: Error computing ylim for {metric_key}: {e}")
            computed_ylim = None
    else:
        computed_ylim = ylim

    # 设置子图属性
    setup_subplot_client(ax, title, time_xLabel, ylim=computed_ylim)


def plot_result(plot_data):
    with open("results/" + plot_data["filename"], 'r') as f:
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

        qps_with_time = plot_averaged_results(short_results, long_results, plot_data["concurrency"],
                                              plot_data["total_time"], plot_data["filename"],
                                              plot_data["exp"])
        plot_comprehensive_results(sorted_all_results, plot_data["concurrency"], plot_data["total_time"],
                                   plot_data["filename"], plot_data["exp"],
                                   fairness_results, qps_with_time)
    else:
        print("No results found")
        return


if __name__ == "__main__":
    with open("tmp_result/plot_data.json", "r") as f:
        data = json.load(f)

    plot_result(data)

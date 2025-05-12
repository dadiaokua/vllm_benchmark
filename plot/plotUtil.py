def xLabel_time(time):
    """Calculate adjusted time and format QPS with time."""
    for i in reversed(range(len(time))):
        if i == 0:
            time[i] = int(0)
            break
        time[i] = int(time[i] - time[0])
    return [f'{time[i]}' for i in range(len(time))]


def plot_metrics_with_annotations(ax, x_data, y_data, label, color, marker, linewidth=2, xytext=(0, 10), linestyle='-'):
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
    ax.set_xlabel('Time(s)')
    if y_format:
        ax.yaxis.set_major_formatter(y_format)
    if ylim:
        ax.set_ylim(ylim)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)


def setup_subplot_client(ax, title, x_labels, ylim=None):
    """设置子图的通用属性"""
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("Time", fontsize=10)
    ax.set_ylabel(title, fontsize=10)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.tick_params(axis='both', labelsize=9)
    ax.grid(True, linestyle='--', alpha=0.7)

    # 检查ylim是否为有效的数值元组
    if ylim is not None and isinstance(ylim, tuple) and len(ylim) == 2:
        try:
            bottom, top = float(ylim[0]), float(ylim[1])
            if bottom < top:  # 确保下限小于上限
                ax.set_ylim(bottom, top)
        except (TypeError, ValueError):
            print(f"Warning: Invalid ylim value: {ylim}. Skipping ylim setting.")


def plot_fairness_index(ax, f_values, times):
    """绘制Jain's公平性指数"""
    if not f_values or not times:
        print("Warning: Empty fairness values or times")
        return
        
    # 确保长度匹配
    min_len = min(len(times), len(f_values))
    times = times[:min_len]
    f_values = f_values[:min_len]
    
    # 绘制公平性指数
    line, = ax.plot(times, f_values, marker='o', color='#1f77b4', linestyle='-', linewidth=2)
    
    # 添加数值标注
    for i, (t, v) in enumerate(zip(times, f_values)):
        if i % 3 == 0 or i == len(f_values) - 1:  # 每隔3个点标注一次
            ax.annotate(f'{v:.3f}',  # 增加小数位数以显示更精确的公平性指数
                       (t, v),
                       xytext=(0, 10),
                       textcoords='offset points',
                       ha='center',
                       va='bottom',
                       fontsize=8)
    
    ax.set_title("Jain's Fairness Index")
    min_f = min(f_values) if f_values else 0.7  # Get minimum fairness value, default to 0.7 if empty
    ax.set_ylim(min_f * 0.7, 1.01)  # 调整y轴范围以更好地显示接近1的值
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlabel("Time Point")
    ax.set_ylabel("Fairness Index")

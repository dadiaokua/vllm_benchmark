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
    ax.set_xticklabels(x_labels, rotation=45)
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
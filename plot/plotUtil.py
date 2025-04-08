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

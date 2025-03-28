import json
import os
import time


def save_results(f_result, s_result, RESULTS_FILE):
    """将公平性结果追加写入 JSON 文件"""
    current_time = time.localtime()

    new_entry = {
        "f_result": f_result,
        "s_result": s_result,
        "time": current_time
    }

    # 读取现有数据
    with open(RESULTS_FILE, "r") as f:
        data = json.load(f)

    # 追加新数据
    data.append(new_entry)

    # **写回文件**
    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f, indent=4)

    print(f"结果已写入 {RESULTS_FILE}: {new_entry}")

def save_benchmark_results(filename, benchmark_results, plot_data):
    """Save benchmark results to files"""
    os.makedirs("results", exist_ok=True)
    print(f"Saving benchmark results to: results/{filename}")

    with open("results/" + filename, 'w') as f:
        json.dump(benchmark_results, f, indent=2)

    with open("tmp_result/plot_data.json", "w") as f:
        json.dump(plot_data, f, indent=4)
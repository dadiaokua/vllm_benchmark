import json
import os
import time


def save_results(exchange_count, f_result, s_result, RESULTS_FILE):
    """将公平性结果追加写入 JSON 文件"""
    # Get current time and format it as YYYY-MM-DD HH:MM:SS (24-hour format)
    current_time_struct = time.localtime()
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", current_time_struct)

    new_entry = {
        "f_result": f_result,
        "s_result": s_result,
        "time": formatted_time,  # Use formatted time string
        "exchange_count": exchange_count
    }

    data = []
    # Ensure the directory exists if RESULTS_FILE includes a path
    results_dir = os.path.dirname(RESULTS_FILE)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)

    # 读取现有数据 (Robust reading)
    try:
        # Check if file exists and is not empty
        if os.path.exists(RESULTS_FILE) and os.path.getsize(RESULTS_FILE) > 0:
            with open(RESULTS_FILE, "r", encoding='utf-8') as f:
                data = json.load(f)
                # Ensure data is a list, otherwise start fresh
                if not isinstance(data, list):
                    print(f"Warning: Content of {RESULTS_FILE} is not a list. Initializing a new list.")
                    data = []
        # If file doesn't exist or is empty, data remains []
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from {RESULTS_FILE}. Initializing a new list.")
        data = [] # Reset data to empty list if decoding fails
    except IOError as e:
        print(f"Error reading file {RESULTS_FILE}: {e}. Initializing a new list.")
        data = [] # Reset data if reading fails

    # 追加新数据
    data.append(new_entry)

    # **写回文件**
    try:
        with open(RESULTS_FILE, "w", encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False) # ensure_ascii=False for non-ASCII chars if any
    except IOError as e:
        print(f"Error writing results to {RESULTS_FILE}: {e}")
        return # Exit if writing fails

    print(f"结果已写入 {RESULTS_FILE}: {new_entry}")


def save_benchmark_results(filename, benchmark_results, plot_data):
    """Save benchmark results to files"""
    os.makedirs("results", exist_ok=True)
    print(f"Saving benchmark results to: results/{filename}")

    with open("results/" + filename, 'w') as f:
        json.dump(benchmark_results, f, indent=2)

    with open("tmp_result/plot_data.json", "w") as f:
        json.dump(plot_data, f, indent=4)


def save_exchange_record(record, filepath):
    """Save exchange record to file"""

    # Create tmp_result directory if it doesn't exist
    if not os.path.exists('tmp_result'):
        os.makedirs('tmp_result')

    try:
        # Read existing records
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                records = json.load(f)
        else:
            records = []

        # Append new record
        records.append(record)

        # Write back to file
        with open(filepath, 'w') as f:
            json.dump(records, f, indent=2)

    except Exception as e:
        print(f"Error saving exchange record: {e}")


def save_to_file(filename, data):
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(data + "\n")

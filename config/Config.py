from datetime import datetime

GLOBAL_CONFIG = {
    "latency_slo": 5,
    "output_tokens": 256,
    "alpha": 0.7,
    "fairness_ratio_LFS": 0.1,
    "fairness_ratio_VTC": 0.5,
    'ADJUST_SENSITIVITY': 1,
    "b": 1.5,
    "whether_fairness": 1,
    "max_granularity": 10,
    "round_time": 60,
    "monitor_file_time": datetime.now().strftime("%m_%d_%H_%M"),
    "exp_time": 36000,
    "avg_success_rate": 0.9,
    "max_exchange_times": 1,
    "prompt_max_len": 10000,
    "request_model_name": "",
    "buffer_ratio": 0.5
}

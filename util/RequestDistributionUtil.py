import numpy as np


def get_target_time(request_count, rate_lambda, global_start_time, distribution, use_time_data, time_data):
    if use_time_data:
        target_time = time_data[request_count]
    else:
        # 计算这个请求应该在什么时间点发送（基于全局开始时间）
        if distribution == "poisson":
            # 泊松分布：请求之间的间隔遵循指数分布
            intervals = [float(np.random.exponential(1 / rate_lambda)) for _ in range(request_count + 1)]
            target_time = global_start_time + sum(intervals)
        elif distribution == "normal":
            # 正态分布：请求均匀分布，但有小的随机波动
            target_time = global_start_time + (request_count / rate_lambda) + float(np.random.normal(0, 0.01))
        else:
            # 均匀分布：请求基本均匀，但有一定范围的随机性
            jitter = np.random.uniform(-0.1, 0.1) / rate_lambda
            target_time = global_start_time + (request_count / rate_lambda) + jitter

    return target_time
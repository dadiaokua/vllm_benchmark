import asyncio
import json

from util.util import save_results

RESULTS_FILE = 'tmp_result/tmp_fairness_result.json'


async def monitor_results(result_queue, update_event, max_loops, client_count):
    loop_count = 0  # 计数器
    tmp_results = []

    # **程序启动时清空文件**
    with open(RESULTS_FILE, "w") as f:
        json.dump([], f)  # 写入空数组

    while loop_count < (max_loops * client_count + 1):  # 限制循环次数
        await update_event.wait()  # 等待新的更新

        while not result_queue.empty():  # 如果队列里有结果
            result = await result_queue.get()  # 获取队列中的结果
            print(f"Received result: {result}")
            tmp_results.append(result)
            if len(tmp_results) == client_count:
                f_result, s_result = await fairness_result(tmp_results)
                # **追加到文件**
                save_results(f_result, s_result, RESULTS_FILE)
                tmp_results = []
            result_queue.task_done()  # 标记任务完成
        loop_count += 1  # 计数+1
    print("Monitor task exiting: Reached max_loops limit.")


def calculate_Jains_index(service):
    n = len(service)
    if n == 0:
        return 0  # Avoid division by zero

    sum_service = sum(service)
    sum_squares = sum(s ** 2 for s in service)

    j = (sum_service ** 2) / (n * sum_squares)
    return j


async def fairness_result(tmp_results):
    print(f"Fairness result: {tmp_results}")
    service = []
    for result in tmp_results:
        service.append(result["total_input_tokens"] + 2 * result["total_output_tokens"])
    tmp_jains_index = calculate_Jains_index(service)
    return tmp_jains_index, service

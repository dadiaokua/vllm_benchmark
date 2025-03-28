import asyncio
import json

from util.FileSaveUtil import save_results

RESULTS_FILE = 'tmp_result/tmp_fairness_result.json'


async def monitor_results(result_queue, update_event, max_loops, client_count):
    loop_count = 0  # 计数器
    tmp_results = []
    print(f'Starting monitor_results with max_loops={max_loops}, client_count={client_count}')
    print(f'Total expected loop_count = {max_loops * client_count + 1}')

    while loop_count < (max_loops * client_count):  # 限制循环次数
        print(f'Current loop count: {loop_count}')
        print('Waiting for update event...')
        await update_event.wait()  # 等待新的更新
        print('Update event received')

        # 重要：清除事件状态，这样下次循环会真正等待
        update_event.clear()

        while not result_queue.empty():  # 如果队列里有结果
            print(f'Queue not empty, getting next result...')
            result = await result_queue.get()  # 获取队列中的结果
            print(f"Received result: {result}")
            tmp_results.append(result)
            loop_count += 1  # 计数+1
            print(f'Completed loop {loop_count}')
            print(f'Current tmp_results length: {len(tmp_results)}')

            if len(tmp_results) == client_count:
                print(f'Reached client_count={client_count}, calculating fairness...')
                f_result, s_result = await fairness_result(tmp_results)
                # **追加到文件**
                save_results(f_result, s_result, RESULTS_FILE)
                print(f'Results saved to {RESULTS_FILE}')
                tmp_results = []
                print('Cleared tmp_results')

            result_queue.task_done()  # 标记任务完成
            print('Task marked as done')

    print("Monitor task exiting: Reached max_loops limit.")
    print(f"Final loop count: {loop_count}")


def calculate_Jains_index(service_list):
    service = [entry["service"] for entry in service_list]
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
        service.append({"service": result["total_input_tokens"] + 2 * result["total_output_tokens"],
                        "client": result["client_index"]})
    tmp_jains_index = calculate_Jains_index(service)
    return tmp_jains_index, service

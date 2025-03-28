import asyncio
import json

from util.FileSaveUtil import save_results
from util.MathUtil import fairness_result

RESULTS_FILE = 'tmp_result/tmp_fairness_result.json'


async def monitor_results(clients, result_queue, update_event, max_loops, client_count):
    loop_count = 0  # 计数器
    tmp_results = []
    print(f'Starting monitor_results with max_loops={max_loops}, client_count={client_count}')
    print(f'Total expected loop_count = {max_loops * client_count}')

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
            tmp_results.append(result)
            loop_count += 1  # 计数+1
            print(f'Completed loop {loop_count}')
            print(f'Current completed client numbers: {len(tmp_results)}')

            if len(tmp_results) == client_count:
                print(f'Reached client_count={client_count}, calculating fairness...')
                f_result, s_result = await fairness_result(clients)
                # **追加到文件**
                save_results(f_result, s_result, RESULTS_FILE)
                print(f'Results saved to {RESULTS_FILE}')
                tmp_results = []
                print('Cleared tmp_results')

            result_queue.task_done()  # 标记任务完成
            print('Task marked as done')

    print("Monitor task exiting: Reached max_loops limit.")
    print(f"Final loop count: {loop_count}")




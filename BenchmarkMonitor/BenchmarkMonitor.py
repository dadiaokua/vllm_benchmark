import asyncio
import json
from datetime import datetime, timedelta

from config.Config import GLOBAL_CONFIG
from util.FileSaveUtil import save_results
from util.MathUtil import fairness_result, is_fairness_LFSLLM, is_fairness_VTC, is_fairness_DLPM

RESULTS_FILE = 'tmp_result/tmp_fairness_result.json'


async def monitor_results(clients, result_queue, client_count, exp_type):
    tmp_results = []
    print(
        f'Starting monitor_results, client_count={client_count}')
    now_time = datetime.now()
    exp_duration = timedelta(seconds=GLOBAL_CONFIG['exp_time'])

    while datetime.now() - now_time < exp_duration:  # 限制循环次数

        # 每秒检查一次队列，而不是等待事件
        if not result_queue.empty():  # 如果队列里有结果
            print(f'Queue not empty, getting next result...')
            try:
                result = await asyncio.wait_for(result_queue.get(), timeout=10)  # 10秒超时
            except asyncio.TimeoutError:
                print("Timeout while waiting for result.")
                await asyncio.sleep(1)  # 如果超时，等待1秒后继续
                continue
                
            tmp_results.append(result)
            print(f'Current completed client numbers: {len(tmp_results)}')

            if len(tmp_results) == client_count:
                print(f'Reached client_count={client_count}, calculating fairness...')
                print("Starting fairness calculation...")
                f_result, s_result = await fairness_result(clients)
                print(f"Fairness calculation complete. Fairness index: {f_result}")

                if GLOBAL_CONFIG["whether_fairness"]:
                    print("Starting fairness adjustment...")
                    if exp_type == "LFS":
                        await is_fairness_LFSLLM(clients, exp_type)
                    elif exp_type == "VTC":
                        await is_fairness_VTC(clients, exp_type)
                    elif exp_type == "DLPM":
                        await is_fairness_DLPM(clients, exp_type)
                    else:
                        print(f"Invalid experiment type: {exp_type}, skipping fairness")
                    print("Fairness adjustment complete")
                else:
                    print("Skipping fairness adjustment (disabled in config)")

                save_results(f_result, s_result, RESULTS_FILE)
                print(f'Results saved to {RESULTS_FILE}')
                tmp_results = []

                print("Notifying clients of completion...")
                for i, client in enumerate(clients):
                    print(f"Resetting client {i + 1}/{len(clients)}")
                    client.exchange_Resources_Times = 0
                    client.monitor_done_event.set()
                print("All clients notified")

            result_queue.task_done()  # 标记任务完成
            print('Task marked as done')
        else:
            print('Queue is empty, waiting for 5 second..., len of tmp_results: ', len(tmp_results))
            # 如果队列为空，等待1秒后再次检查
            await asyncio.sleep(5)


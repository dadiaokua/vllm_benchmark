import unittest
from unittest.mock import Mock, patch, AsyncMock
import asyncio
import time
import numpy as np
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import ChoiceDelta

from util.RequestUtil import make_request, process_request, worker, calculate_raw_request_time, get_target_time


class TestRequestUtil(unittest.TestCase):
    def setUp(self):
        # 设置通用的测试数据
        self.tokenizer = Mock()
        self.tokenizer.return_value.input_ids = [Mock(return_value=[1, 2, 3])]
        self.client = AsyncMock()
        self.output_tokens = 50
        self.request_timeout = 10
        self.latency_slo = 5

    async def async_setup(self):
        # 异步设置，如果需要的话
        pass

    @patch('time.time')
    async def test_make_request(self, mock_time):
        # 模拟时间
        mock_time.side_effect = [0, 1, 2]  # start_time, first_token_time, end_time

        # 模拟流式响应
        mock_chunk = ChatCompletionChunk(
            id="123",
            choices=[Choice(delta=ChoiceDelta(content="test"), finish_reason=None, index=0)],
            created=123,
            model="test",
            object="chat.completion.chunk"
        )

        # 设置异步模拟响应
        self.client.chat.completions.create.return_value.__aiter__.return_value = [mock_chunk]

        result = await make_request(
            self.client,
            self.output_tokens,
            self.request_timeout,
            "test request",
            self.tokenizer,
            self.latency_slo
        )

        # 验证结果
        self.assertIsNotNone(result)
        total_tokens, elapsed_time, tokens_per_second, ttft, input_tokens, slo_violation = result
        self.assertEqual(total_tokens, 1)
        self.assertEqual(elapsed_time, 2)
        self.assertEqual(ttft, 1)

    def test_calculate_raw_request_time(self):
        # 测试不同分布的时间计算
        distributions = ["poisson", "normal", "uniform"]
        rate_lambda = 10
        global_start_time = time.time()
        request_count = 5

        for dist in distributions:
            time1 = calculate_raw_request_time(request_count, rate_lambda, global_start_time, dist)
            time2 = calculate_raw_request_time(request_count + 1, rate_lambda, global_start_time, dist)

            # 验证时间间隔合理性
            self.assertGreater(time2, time1)
            interval = time2 - time1
            self.assertGreater(interval, 0)
            self.assertLess(abs(interval - 1 / rate_lambda), 0.5)  # 允许0.5秒的误差

    def test_get_target_time(self):
        # 测试不同参数组合的目标时间计算
        test_cases = [
            {
                "rate_lambda": 1,
                "active_ratio": 1,
                "time_ratio": 1,
                "window_length": 10
            },
            {
                "rate_lambda": 2,
                "active_ratio": 0.5,
                "time_ratio": 2,
                "window_length": 20
            }
        ]

        for case in test_cases:
            time1 = get_target_time(
                request_count=0,
                rate_lambda=case["rate_lambda"],
                global_start_time=time.time(),
                distribution="uniform",
                use_time_data=False,
                time_data=None,
                active_ratio=case["active_ratio"],
                window_length=case["window_length"],
                time_ratio=case["time_ratio"]
            )

            self.assertIsNotNone(time1)
            # 添加更多具体的验证...

    @patch('asyncio.sleep', new_callable=AsyncMock)
    async def test_worker(self, mock_sleep):
        # 测试worker的基本功能
        selected_clients = [AsyncMock()]
        semaphore = asyncio.Semaphore(1)
        results = []
        client_index = "test_client"
        rate_lambda = 1
        sample_content = ["test request"]
        config_round = 0
        worker_id = 0
        round_time = 2  # 测试2秒的运行时间

        request_count, drift_time = await worker(
            selected_clients=selected_clients,
            semaphore=semaphore,
            results=results,
            output_tokens=self.output_tokens,
            client_index=client_index,
            tokenizer=self.tokenizer,
            request_timeout=self.request_timeout,
            round_time=round_time,
            rate_lambda=rate_lambda,
            distribution="uniform",
            sample_content=sample_content,
            config_round=config_round,
            worker_id=worker_id,
            time_data=None,
            use_time_data=False,
            latency_slo=self.latency_slo,
            active_ratio=1,
            time_ratio=1
        )

        # 验证结果
        self.assertGreater(request_count, 0)
        self.assertEqual(drift_time, 0)
        # 验证sleep被调用
        mock_sleep.assert_called()


if __name__ == '__main__':
    unittest.main()

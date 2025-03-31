import numpy as np
from openai import AsyncOpenAI
from sympy import symbols, Eq, solve

from config.Config import GLOBAL_CONFIG


def initialize_clients(local_port):
    """Initialize OpenAI clients based on port configuration"""
    if isinstance(local_port, list):
        return [AsyncOpenAI(base_url=f"http://localhost:{port}/v1") for port in local_port]
    else:
        return [AsyncOpenAI(base_url=f"http://localhost:{local_port}/v1")]


def ExchangeQPS(client1, client2):
    print(client1.service / client1.avg_latency_div_standard_latency)
    x = symbols('x')
    equation = Eq(((client1.service - x) / client1.avg_latency_div_standard_latency) / (
        ((client2.service + x) / client2.avg_latency_div_standard_latency)), GLOBAL_CONFIG["a"])
    serviceX = solve(equation, x)
    if serviceX > 0.1 * client1.service:
        tmp = client1.qps * 0.1
        client1.qps = client1.qps - tmp
        client2.qps = client2.qps + tmp
    else:
        tmp = client1.qps * serviceX / client1.service
        client1.qps = client1.qps - tmp
        client2.qps = client2.qps + tmp

import numpy as np
from openai import AsyncOpenAI



def initialize_clients(local_port):
    """Initialize OpenAI clients based on port configuration"""
    if isinstance(local_port, list):
        return [AsyncOpenAI(base_url=f"http://localhost:{port}/v1") for port in local_port]
    else:
        return [AsyncOpenAI(base_url=f"http://localhost:{local_port}/v1")]

def generate_configurations(args):
    """Generate benchmark configurations"""
    short_configurations = []
    long_configurations = []

    for qps_value in np.arange(args.range_lower, args.range_higher + 1.1, 1):
        if args.short_qps == 0 and args.long_qps == 0:
            print("[Error]: short_qps and long_qps cannot be 0 at the same time")
            exit(1)

        qps_value = round(qps_value, 1)

        if args.short_qps == 0:
            if abs(qps_value % args.long_qps) < 1e-6:
                configuration = {"num_requests": args.num_requests, "qps": args.range_lower, "output_tokens": 200}
                short_configurations.append(configuration)
        else:
            if abs(qps_value % args.short_qps) < 1e-6:
                configuration = {"num_requests": args.num_requests, "qps": int(qps_value), "output_tokens": 200}
                short_configurations.append(configuration)

        if args.long_qps == 0:
            if abs(qps_value % args.short_qps) < 1e-6:
                configuration = {"num_requests": args.num_requests, "qps": args.range_lower, "output_tokens": 200}
                long_configurations.append(configuration)
        else:
            if abs(qps_value % args.long_qps) < 1e-6:
                configuration = {"num_requests": args.num_requests, "qps": int(qps_value), "output_tokens": 200}
                long_configurations.append(configuration)

    return short_configurations, long_configurations
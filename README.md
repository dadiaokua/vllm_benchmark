# vLLM Benchmark

This repository contains scripts for benchmarking the performance of large language models (LLMs) served using vLLM. It's designed to test the scalability and performance of LLM deployments under various concurrency levels.

## Features

- **Automatic vLLM Engine Startup**: Built-in vLLM engine management with configurable parameters
- Benchmark LLMs with different concurrency levels
- Measure key performance metrics:
  - Requests per second
  - Latency
  - Tokens per second
  - Time to first token
- Easy to run with customizable parameters
- Generates JSON output for further analysis or visualization

## Requirements

- Python 3.7+
- `openai` Python package
- `numpy` Python package
- `vllm` package (for engine startup)
- `requests` package

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/vllm-benchmark.git
   cd vllm-benchmark
   ```

2. Install the required packages:
   ```
   pip install openai numpy vllm requests
   ```

## vLLM Engine Management

The benchmark system supports two modes for using vLLM:

### Mode 1: AsyncLLMEngine Direct API (Recommended)

Uses vLLM's AsyncLLMEngine API directly in Python for better performance and resource efficiency:

```bash
python run_benchmarks.py \
  --start_engine True \
  --model_path "/home/llm/model_hub/Qwen2.5-32B-Instruct" \
  --tensor_parallel_size 8 \
  --gpu_memory_utilization 0.9 \
  --vllm_url "http://127.0.0.1:8000" \  # Still needed for compatibility, but not used for requests
  --api_key "test" \
  # ... other benchmark parameters
```

**Benefits:**
- **Direct Integration**: No HTTP overhead, direct Python API calls
- **Better Performance**: Faster than HTTP requests, no serialization overhead
- **Resource Efficiency**: Shares the same Python process memory space
- **Easier Debugging**: All components run in the same process

### Mode 2: HTTP Server Mode (Traditional)

Uses an external vLLM HTTP server via OpenAI-compatible API:

```bash
# First start vLLM server separately (or use --start_engine False if already running)
python run_benchmarks.py \
  --start_engine False \
  --vllm_url "http://existing-server:8000" \
  --api_key "test" \
  # ... other benchmark parameters
```

### Automatic Mode Detection

The system automatically detects which mode to use:
- If `--start_engine True`: Uses AsyncLLMEngine direct API
- If `--start_engine False`: Uses HTTP client to connect to external server

### Available vLLM Engine Parameters

- `--start_engine`: Whether to start the vLLM engine (default: True)
- `--model_path`: Path to the vLLM model (default: "/home/llm/model_hub/Qwen2.5-32B-Instruct")
- `--tensor_parallel_size`: Tensor parallel size (default: 8)
- `--pipeline_parallel_size`: Pipeline parallel size (default: 1)
- `--gpu_memory_utilization`: GPU memory utilization (default: 0.9)
- `--max_model_len`: Maximum model length (default: 8124)
- `--max_num_seqs`: Maximum number of sequences (default: 256)
- `--max_num_batched_tokens`: Maximum number of batched tokens (default: 65536)
- `--swap_space`: Swap space size in GB (default: 4)
- `--device`: Device type (default: "cuda")
- `--dtype`: Data type (default: "float16")
- `--quantization`: Quantization method (default: "None")
- `--trust_remote_code`: Trust remote code (default: True)
- `--enable_chunked_prefill`: Enable chunked prefill (default: False)
- `--disable_log_stats`: Disable log statistics (default: False)

### Sampling Parameters (AsyncLLMEngine Mode Only)

When using AsyncLLMEngine direct API mode, you can configure sampling parameters in `config/Config.py`:

```python
GLOBAL_CONFIG = {
    # ... other config ...
    
    # AsyncLLMEngine采样参数
    "sampling_temperature": 0.7,      # Controls randomness (0.0 = deterministic, 1.0+ = more random)
    "sampling_top_p": 0.9,            # Nucleus sampling threshold
    "sampling_top_k": -1,             # Top-k sampling (-1 = disabled)
    "sampling_repetition_penalty": 1.0, # Repetition penalty (1.0 = no penalty)
}
```

## Usage

### Single Benchmark Run

To run a single benchmark:

```
python vllm_benchmark.py --num_requests 100 --concurrency 10 --output_tokens 100 --vllm_url "http://localhost:8000/v1" --api_key "your-api-key"
```

Parameters:
- `num_requests`: Total number of requests to make
- `concurrency`: Number of concurrent requests
- `output_tokens`: Number of tokens to generate per request
- `vllm_url`: URL of the vLLM server
- `api_key`: API key for the vLLM server
- `request_timeout`: (Optional) Timeout for each request in seconds (default: 30)

### Multiple Benchmark Runs with Auto Engine Startup

To run benchmarks with automatic vLLM engine startup:

```bash
python run_benchmarks.py \
  --model_path "/home/llm/model_hub/Qwen2.5-32B-Instruct" \
  --vllm_url "http://127.0.0.1:8000" \
  --api_key "test" \
  --short_qpm "60" \
  --long_qpm "60" \
  --short_clients 1 \
  --long_clients 1 \
  --short_clients_slo "10" \
  --long_clients_slo "10" \
  --round 5 \
  --round_time 300 \
  --exp "LFS" \
  --distribution "poisson" \
  --local_port 8000 \
  --remote_port 8000 \
  --use_tunnel 0
```

This will:
1. Automatically start a vLLM engine with the specified model
2. Wait for the engine to be ready
3. Run the benchmark experiments
4. Automatically stop the engine when finished

## Output

The benchmark results are saved in JSON format, containing detailed metrics for each run, including:

- Total requests and successful requests
- Requests per second
- Total output tokens
- Latency (average, p50, p95, p99)
- Tokens per second (average, p50, p95, p99)
- Time to first token (average, p50, p95, p99)

## Results

Please see the results directory for benchmarks on [Backprop](https://backprop.co) instances.

## Contributing

Contributions to improve the benchmarking scripts or add new features are welcome! Please feel free to submit pull requests or open issues for any bugs or feature requests.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

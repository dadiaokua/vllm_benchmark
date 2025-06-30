# vLLM Benchmark

This repository contains scripts for benchmarking the performance of large language models (LLMs) served using vLLM. It's designed to test the scalability and performance of LLM deployments under various concurrency levels and scheduling strategies.

## Features

- **Automatic vLLM Engine Startup**: Built-in vLLM engine management with configurable parameters
- **Multi-Experiment Support**: Run multiple scheduling experiments in sequence with a single command
- **Multiple Scheduling Strategies**: LFS, VTC, FCFS, and Queue-based strategies
- **Easy Configuration**: Bash script with command-line argument support
- Benchmark LLMs with different concurrency levels
- Measure key performance metrics:
  - Requests per second
  - Latency
  - Tokens per second
  - Time to first token
  - Fairness metrics (Jain's Index)
- Advanced plotting capabilities with separate views for performance, fairness, and aggregated metrics
- Generates JSON output for further analysis or visualization

## Requirements

- Python 3.7+
- `openai` Python package
- `numpy` Python package
- `vllm` package (for engine startup)
- `requests` package
- `matplotlib` (for plotting)
- `transformers` (for tokenizer)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/vllm-benchmark.git
   cd vllm-benchmark
   ```

2. Install the required packages:
   ```bash
   pip install openai numpy vllm requests matplotlib transformers
   ```

## Quick Start

The easiest way to run benchmarks is using the provided bash script:

### Single Experiment

```bash
# Run LFS experiment
./start_vllm_benchmark.sh -e LFS

# Run VTC experiment  
./start_vllm_benchmark.sh -e VTC

# Run queue-based LFS experiment
./start_vllm_benchmark.sh -e QUEUE_LFS
```

### Multiple Experiments (Batch Mode)

```bash
# Compare different scheduling strategies
./start_vllm_benchmark.sh -e LFS -e VTC -e FCFS

# Test all queue-based strategies
./start_vllm_benchmark.sh -e QUEUE_LFS -e QUEUE_VTC -e QUEUE_FCFS -e QUEUE_ROUND_ROBIN

# Mixed comparison
./start_vllm_benchmark.sh -e LFS --exp QUEUE_LFS -e VTC
```

### Available Experiment Types

| Experiment Type | Description |
|----------------|-------------|
| `LFS` | Least Fair Share - Direct scheduling |
| `VTC` | Virtual Time Credits - Direct scheduling |
| `FCFS` | First Come First Serve - Direct scheduling |
| `QUEUE_LFS` | Queue-based LFS scheduling |
| `QUEUE_VTC` | Queue-based VTC scheduling |
| `QUEUE_FCFS` | Queue-based FCFS scheduling |
| `QUEUE_ROUND_ROBIN` | Queue-based Round Robin scheduling |
| `QUEUE_SJF` | Queue-based Shortest Job First |
| `QUEUE_FAIR` | Queue-based Fair Share scheduling |

### Getting Help

```bash
./start_vllm_benchmark.sh -h
# or
./start_vllm_benchmark.sh --help
```

## Configuration

The bash script (`start_vllm_benchmark.sh`) contains pre-configured parameters that you can modify:

### vLLM Engine Parameters
- Model path and tensor parallel configuration
- GPU memory utilization and sequence limits
- Data type and quantization settings

### Client Configuration
- Short clients: 7 clients with QPS range 50-150
- Long clients: 3 clients with QPS around 50-80
- SLO (Service Level Objectives) settings

### Experiment Settings
- 20 rounds per experiment
- 300 seconds per round
- Configurable through script variables

## Advanced Usage

### Manual Python Execution

For more control, you can run the Python script directly:

```bash
cd run_benchmark
python3 run_benchmarks.py \
    --vllm_url "http://localhost:8000/v1" \
    --api_key "test" \
    --exp "LFS" \
    --short_clients 7 \
    --long_clients 3 \
    --round 20 \
    --round_time 300 \
    # ... other parameters
```

### Plotting Results

The system automatically generates three types of plots:

1. **Performance Metrics**: Individual client performance over time
2. **Fairness Metrics**: Fairness ratios, Jain's index, and credits
3. **Aggregated Metrics**: System-wide performance summaries

Plots are saved in the `figure/` directory with timestamps.

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

## Batch Execution Features

### Sequential Execution
When running multiple experiments:
- Experiments run sequentially, one after another
- Each experiment completes fully before the next begins
- Automatic progress tracking with experiment counters

### Status Reporting
```bash
🚀🚀🚀 开始执行实验 1/3: LFS 🚀🚀🚀
[Experiment execution...]
✅ 实验 1/3: LFS 已完成

📋 准备开始下一个实验: 2/3 - VTC
==========================================
```

### Final Summary
```bash
🎉🎉🎉 所有实验执行完成！🎉🎉🎉
==========================================
📊 执行结果总览:
  总实验数: 3
  成功实验数: 3
  失败实验数: 0

✅ 成功的实验:
    - LFS
    - VTC
    - FCFS
```

## Log Management

The system generates timestamped log files for each run:

```
log/
├── client_short_20250101_143022.log    # Short client logs
├── client_long_20250101_143022.log     # Long client logs
├── monitor_LFS_20250101_143022.log     # Monitor logs
└── run_benchmarks_20250101_143022.log  # Main program logs
```

Each experiment run creates a new set of log files, preserving history while keeping current runs separate.

## Output and Results

### JSON Results
Results are saved in timestamped JSON files in the `results/` directory:
- Individual client performance metrics
- Fairness calculations and Jain's index
- System-wide aggregated statistics

### Plots
Three types of plots are automatically generated:
1. **Performance plots** (`performance_metrics_*.png`)
2. **Fairness plots** (`fairness_metrics_*.png`)  
3. **Aggregated plots** (`aggregated_metrics_*.png`)

### Metrics Included
- **Performance**: Latency, tokens/sec, requests/sec, success rate
- **Fairness**: Jain's fairness index, client fairness ratios, credit systems
- **System**: Total throughput, average performance, SLO violations

## Contributing

Contributions to improve the benchmarking scripts or add new features are welcome! Please feel free to submit pull requests or open issues for any bugs or feature requests.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

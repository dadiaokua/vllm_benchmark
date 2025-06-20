#!/bin/bash

# =============================================================================
# vLLM Benchmark 启动脚本 (简化版本)
# 只包含基准测试参数，不包含vLLM引擎参数
# =============================================================================

# ========== 基础连接参数 ==========
VLLM_URL="http://222.201.144.119:8000/v1"
API_KEY="test"
USE_TUNNEL=0
LOCAL_PORT="8000"
REMOTE_PORT="10085"

# ========== 请求配置参数 ==========
DISTRIBUTION="normal"
SHORT_QPM="50 50 60 60 80 100 150"
SHORT_CLIENT_QPM_RATIO=1
LONG_QPM="50 50 80"
LONG_CLIENT_QPM_RATIO=1

# ========== 客户端配置参数 ==========
SHORT_CLIENTS=7
SHORT_CLIENTS_SLO="6 7 8 9 10 11 12"
LONG_CLIENTS=3
LONG_CLIENTS_SLO="7 9 11"

# ========== 并发和性能参数 ==========
CONCURRENCY=5
NUM_REQUESTS=100
REQUEST_TIMEOUT=30
SLEEP_TIME=1

# ========== 实验配置参数 ==========
ROUND_NUM=20
ROUND_TIME=300
EXP_NAME="LFS"
USE_TIME_DATA=0

# ========== 模型和tokenizer参数 ==========
TOKENIZER_PATH="/Users/myrick/modelHub/hub/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
REQUEST_MODEL_NAME="Meta-Llama-3.1-8B-Instruct-AWQ-INT4"

# ========== 显示配置信息 ==========
echo "=========================================="
echo "         vLLM Benchmark 配置信息"
echo "=========================================="
echo ""
echo "🔗 基础连接参数:"
echo "  vLLM URL: $VLLM_URL"
echo "  API Key: $API_KEY"
echo "  Use Tunnel: $USE_TUNNEL"
echo "  Local Port: $LOCAL_PORT"
echo "  Remote Port: $REMOTE_PORT"
echo ""
echo "📊 请求配置参数:"
echo "  Distribution: $DISTRIBUTION"
echo "  Short QPM: $SHORT_QPM"
echo "  Short Client QPM Ratio: $SHORT_CLIENT_QPM_RATIO"
echo "  Long QPM: $LONG_QPM"
echo "  Long Client QPM Ratio: $LONG_CLIENT_QPM_RATIO"
echo ""
echo "👥 客户端配置参数:"
echo "  Short Clients: $SHORT_CLIENTS"
echo "  Short Clients SLO: $SHORT_CLIENTS_SLO"
echo "  Long Clients: $LONG_CLIENTS"
echo "  Long Clients SLO: $LONG_CLIENTS_SLO"
echo ""
echo "⚡ 并发和性能参数:"
echo "  Concurrency: $CONCURRENCY"
echo "  Num Requests: $NUM_REQUESTS"
echo "  Request Timeout: $REQUEST_TIMEOUT"
echo "  Sleep Time: $SLEEP_TIME"
echo ""
echo "🧪 实验配置参数:"
echo "  Round Num: $ROUND_NUM"
echo "  Round Time: $ROUND_TIME"
echo "  Experiment Name: $EXP_NAME"
echo "  Use Time Data: $USE_TIME_DATA"
echo ""
echo "🤖 模型和tokenizer参数:"
echo "  Tokenizer Path: $TOKENIZER_PATH"
echo "  Request Model Name: $REQUEST_MODEL_NAME"
echo ""
echo "=========================================="
echo "正在启动基准测试..."
echo "=========================================="

# ========== 启动基准测试 ==========
cd run_benchmark

python3 run_benchmarks.py \
    --vllm_url "$VLLM_URL" \
    --api_key "$API_KEY" \
    --use_tunnel "$USE_TUNNEL" \
    --local_port "$LOCAL_PORT" \
    --remote_port "$REMOTE_PORT" \
    --distribution "$DISTRIBUTION" \
    --short_qpm "$SHORT_QPM" \
    --short_client_qpm_ratio "$SHORT_CLIENT_QPM_RATIO" \
    --long_qpm "$LONG_QPM" \
    --long_client_qpm_ratio "$LONG_CLIENT_QPM_RATIO" \
    --short_clients "$SHORT_CLIENTS" \
    --short_clients_slo "$SHORT_CLIENTS_SLO" \
    --long_clients "$LONG_CLIENTS" \
    --long_clients_slo "$LONG_CLIENTS_SLO" \
    --concurrency "$CONCURRENCY" \
    --num_requests "$NUM_REQUESTS" \
    --request_timeout "$REQUEST_TIMEOUT" \
    --sleep "$SLEEP_TIME" \
    --round "$ROUND_NUM" \
    --round_time "$ROUND_TIME" \
    --exp "$EXP_NAME" \
    --use_time_data "$USE_TIME_DATA" \
    --tokenizer "$TOKENIZER_PATH" \
    --request_model_name "$REQUEST_MODEL_NAME"

echo ""
echo "=========================================="
echo "基准测试完成！"
echo "=========================================="

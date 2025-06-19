#!/bin/bash

# 默认参数
VLLM_URL="http://222.201.144.119:8000/v1"
API_KEY="http://222.201.144.120:8000/v1" # 注意：脚本中参数名为api_key，但通常API Key本身不是一个URL，请确认其含义
REQUEST_TIMEOUT=30
SHORT_QPS="50 50 60 60 80 100 150"
SHORT_CLIENT_QPS_RATIO=1
LONG_QPS="50 50 80"
LONG_CLIENT_QPS_RATIO=1
DISTRIBUTION="normal"
CONCURRENCY=5
NUM_REQUESTS=100
SHORT_CLIENTS=7
SHORT_CLIENTS_SLO="6 7 8 9 10 11 12"
LONG_CLIENTS=3
LONG_CLIENTS_SLO="7 9 11"
SLEEP_TIME=1 # 原参数为 --sleep，这里用 SLEEP_TIME
LOCAL_PORT=8000
REMOTE_PORT=10085
USE_TIME_DATA=0
ROUND_NUM=20 # 原参数为 --round，这里用 ROUND_NUM
ROUND_TIME=300
EXP_NAME="QUEUE_LFS" # 原参数为 --exp，这里用 EXP_NAME
USE_TUNNEL=0
TOKENIZER_PATH="/home/llm/model_hub/Qwen2.5-32B-Instruct"
MODEL_NAME="Qwen2.5-32B"

# 从命令行读取参数
# u: vllm_url, k: api_key, t: request_timeout, a: short_qps, b: short_client_qps_ratio
# c: long_qps, d: long_client_qps_ratio, f: distribution, g: concurrency, n: num_requests
# H: short_clients, I: short_clients_slo, J: long_clients, K: long_clients_slo
# l: sleep_time, P: local_port, R: remote_port, T: use_time_data
# N: round_num, M: round_time, E: exp_name, U: use_tunnel, Z: tokenizer_path
while getopts u:k:t:a:b:c:d:f:g:n:H:I:J:K:l:P:R:T:N:M:E:U:Z:m: flag
do
    case "${flag}" in
        u) VLLM_URL=${OPTARG};;
        k) API_KEY=${OPTARG};;
        t) REQUEST_TIMEOUT=${OPTARG};;
        a) SHORT_QPS=${OPTARG};;
        b) SHORT_CLIENT_QPS_RATIO=${OPTARG};;
        c) LONG_QPS=${OPTARG};;
        d) LONG_CLIENT_QPS_RATIO=${OPTARG};;
        f) DISTRIBUTION=${OPTARG};;
        g) CONCURRENCY=${OPTARG};;
        n) NUM_REQUESTS=${OPTARG};;
        H) SHORT_CLIENTS=${OPTARG};;
        I) SHORT_CLIENTS_SLO=${OPTARG};;
        J) LONG_CLIENTS=${OPTARG};;
        K) LONG_CLIENTS_SLO=${OPTARG};;
        l) SLEEP_TIME=${OPTARG};;
        P) LOCAL_PORT=${OPTARG};;
        R) REMOTE_PORT=${OPTARG};;
        T) USE_TIME_DATA=${OPTARG};;
        N) ROUND_NUM=${OPTARG};;
        M) ROUND_TIME=${OPTARG};;
        E) EXP_NAME=${OPTARG};;
        U) USE_TUNNEL=${OPTARG};;
        Z) TOKENIZER_PATH=${OPTARG};;
	m) MODEL_NAME=${OPTARG};;
    esac
done

echo "--- Starting vLLM Benchmark ---"
echo "vLLM URL: $VLLM_URL"
echo "API Key: $API_KEY" # 再次提醒确认此参数的实际含义和格式
echo "Request Timeout: $REQUEST_TIMEOUT"
echo "Short QPS: $SHORT_QPS"
echo "Short Client QPS Ratio: $SHORT_CLIENT_QPS_RATIO"
echo "Long QPS: $LONG_QPS"
echo "Long Client QPS Ratio: $LONG_CLIENT_QPS_RATIO"
echo "Distribution: $DISTRIBUTION"
echo "Concurrency: $CONCURRENCY"
echo "Num Requests: $NUM_REQUESTS"
echo "Short Clients: $SHORT_CLIENTS"
echo "Short Clients SLO: $SHORT_CLIENTS_SLO"
echo "Long Clients: $LONG_CLIENTS"
echo "Long Clients SLO: $LONG_CLIENTS_SLO"
echo "Sleep Time: $SLEEP_TIME"
echo "Local Port: $LOCAL_PORT"
echo "Remote Port: $REMOTE_PORT"
echo "Use Time Data: $USE_TIME_DATA"
echo "Round Num: $ROUND_NUM"
echo "Round Time: $ROUND_TIME"
echo "Experiment Name: $EXP_NAME"
echo "Use Tunnel: $USE_TUNNEL"
echo "Tokenizer Path: $TOKENIZER_PATH"
echo "Model Name: $MODEL_NAME"
echo "---"

python3 run_benchmarks.py \
    --vllm_url "$VLLM_URL" \
    --api_key "$API_KEY" \
    --request_timeout "$REQUEST_TIMEOUT" \
    --short_qpm "$SHORT_QPS" \
    --short_client_qpm_ratio "$SHORT_CLIENT_QPS_RATIO" \
    --long_qpm "$LONG_QPS" \
    --long_client_qpm_ratio "$LONG_CLIENT_QPS_RATIO" \
    --distribution "$DISTRIBUTION" \
    --concurrency "$CONCURRENCY" \
    --num_requests "$NUM_REQUESTS" \
    --short_clients "$SHORT_CLIENTS" \
    --short_clients_slo "$SHORT_CLIENTS_SLO" \
    --long_clients "$LONG_CLIENTS" \
    --long_clients_slo "$LONG_CLIENTS_SLO" \
    --sleep "$SLEEP_TIME" \
    --local_port "$LOCAL_PORT" \
    --remote_port "$REMOTE_PORT" \
    --use_time_data "$USE_TIME_DATA" \
    --round "$ROUND_NUM" \
    --round_time "$ROUND_TIME" \
    --exp "$EXP_NAME" \
    --use_tunnel "$USE_TUNNEL" \
    --tokenizer "$TOKENIZER_PATH" \
    --request_model_name "$MODEL_NAME"

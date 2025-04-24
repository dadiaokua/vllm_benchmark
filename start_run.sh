#!/bin/bash

python3 run_benchmarks.py \
  --vllm_url http://222.201.144.120:8000/v1 \
  --api_key http://222.201.144.119:8000/v1 \
  --request_timeout 20 \
  --short_qps 5 5 5 5 5 \
  --short_client_qps_ratio 1 \
  --long_qps 50 \
  --long_client_qps_ratio 1 \
  --distribution poisson \
  --concurrency 5 \
  --num_requests 100 \
  --short_clients 5 \
  --long_clients 0 \
  --sleep 5 \
  --local_port 8000 \
  --remote_port 10085 \
  --use_time_data 0 \
  --round 5 \
  --round_time 60 \
  --exp LFS
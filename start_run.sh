#!/bin/bash

python3 run_benchmarks.py \
  --vllm_url http://222.201.144.119:8000/v1 \
  --api_key http://222.201.144.120:8000/v1 \
  --request_timeout 10 \
  --short_qps 10 \
  --short_client_qps_ratio 1 \
  --long_qps 10 \
  --long_client_qps_ratio 1.2 \
  --distribution normal \
  --concurrency 1 \
  --short_clients 3 \
  --short_clients_slo 2 3 4 \
  --long_clients 1 \
  --long_clients_slo 6 \
  --sleep 5 \
  --local_port 8080 \
  --remote_port 10028 \
  --use_time_data 0 \
  --round 20 \
  --round_time 60 \
  --exp LFS

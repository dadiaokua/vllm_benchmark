#!/bin/bash

# 定义服务地址和模型名称
VLLM_API_URL="http://localhost:8000/v1/chat/completions"
MODEL_NAME="Qwen2.5-32B"

# 生成唯一请求ID
REQUEST_ID=$(uuidgen)

echo "发送请求，request_id=$REQUEST_ID"

# 发送请求，后台执行
curl -X POST $VLLM_API_URL \
  -H "Content-Type: application/json" \
  -H "X-Request-Id: $REQUEST_ID" \
  -d '{
        "model": "'$MODEL_NAME'",
        "messages": [{"role": "user", "content": "Hello, cancel me!"}]
      }' &

# 立即取消请求，调用假设的取消接口
echo "尝试取消请求，request_id=$REQUEST_ID"

curl -X POST http://localhost:8000/abort \
  -H "Content-Type: application/json" \
  -d '{"request_id": "'$REQUEST_ID'"}'

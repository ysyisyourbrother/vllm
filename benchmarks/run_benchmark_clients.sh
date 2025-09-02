#!/bin/bash

# 多进程客户端benchmark执行脚本
# 使用方法:
#   1. 修改下面的参数配置
#   2. 执行: ./run_benchmark_clients.sh

# 配置参数 - 请修改这些参数
TOKENIZER_PATH=""                    # 必填: tokenizer路径
DATASET_PATH=""                      # 必填: mooncake数据集路径
NUM_CLIENTS="4"                      # 客户端进程数
DECODER_HOST="localhost"             # decode instance主机
DECODER_PORT="8002"                  # decode instance端口
NUM_PROMPTS="100"                    # 每个客户端的请求数
REQUEST_RATE="inf"                   # 请求速率 (inf=无限制, 或数字如"10")
FIXED_MAX_OUTPUT_TOKENS=""           # 固定的max_output_tokens (可选)

# 检查必填参数
if [ -z "$TOKENIZER_PATH" ] || [ -z "$DATASET_PATH" ]; then
    echo "错误: 请先设置 TOKENIZER_PATH 和 DATASET_PATH"
    echo "使用方法: $0"
    echo "请直接修改脚本中的参数配置"
    exit 1
fi

# 脚本路径
SCRIPT_PATH="vllm/benchmarks/benchmark_client_mock_prefill_and_proxy.py"

# 设置日志文件和种子
SEED=42
LOG_FILE="./logs/benchmark_clients_${NUM_CLIENTS}.log"

echo "[$(date '+%H:%M:%S')] 启动 ${NUM_CLIENTS} 个客户端进程"
echo "每个客户端处理 ${NUM_PROMPTS} 个请求"
echo "总请求数: $((NUM_CLIENTS * NUM_PROMPTS))"
echo "日志文件: ${LOG_FILE}"
mkdir -p ./logs

# 构建Python命令参数
PYTHON_ARGS=(
    --tokenizer "$TOKENIZER_PATH"
    --dataset-path "$DATASET_PATH"
    --decoder-host "$DECODER_HOST"
    --decoder-port "$DECODER_PORT"
    --num-prompts "$NUM_PROMPTS"
    --num-clients "$NUM_CLIENTS"
    --request-rate "$REQUEST_RATE"
    --seed "$SEED"
)

# 如果设置了固定的max_output_tokens，添加到参数中
if [ -n "$FIXED_MAX_OUTPUT_TOKENS" ]; then
    PYTHON_ARGS+=(--fixed-max-output-tokens "$FIXED_MAX_OUTPUT_TOKENS")
fi

echo "[$(date '+%H:%M:%S')] 执行命令: python $SCRIPT_PATH ${PYTHON_ARGS[*]}"

# 执行Python脚本 (使用tee同时输出到终端和日志文件)
python "$SCRIPT_PATH" "${PYTHON_ARGS[@]}" 2>&1 | tee "$LOG_FILE"

echo "[$(date '+%H:%M:%S')] 所有客户端进程执行完成!"
echo "日志文件: ${LOG_FILE}"

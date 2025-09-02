#!/bin/bash

# 单个客户端benchmark执行脚本
# 使用方法:
#   1. 修改下面的参数
#   2. 单个客户端: ./run_benchmark_clients.sh
#   3. 多个客户端并发: 在多个终端同时运行，使用不同的CLIENT_ID

# 配置参数 - 请修改这些参数
TOKENIZER_PATH=""                    # 必填: tokenizer路径
DATASET_PATH=""                      # 必填: mooncake数据集路径
CLIENT_ID="${1:-1}"                  # 客户端ID (从命令行参数获取，默认为1)
DECODER_HOST="localhost"             # decode instance主机
DECODER_PORT="8002"                  # decode instance端口
NUM_PROMPTS="100"                    # 请求数
REQUEST_RATE="inf"                   # 请求速率 (inf=无限制, 或数字如"10")
FIXED_MAX_OUTPUT_TOKENS=""           # 固定的max_output_tokens (可选)

# 检查必填参数
if [ -z "$TOKENIZER_PATH" ] || [ -z "$DATASET_PATH" ]; then
    echo "错误: 请先设置 TOKENIZER_PATH 和 DATASET_PATH"
    echo "使用方法: $0 [CLIENT_ID]"
    echo "示例: $0 1    # 启动客户端1"
    echo "示例: $0 2    # 启动客户端2"
    exit 1
fi

# 脚本路径
SCRIPT_PATH="vllm/benchmarks/benchmark_client_mock_prefill_and_proxy.py"

# 设置日志文件和种子
SEED=$((42 + CLIENT_ID))
LOG_FILE="./logs/benchmark_client_${CLIENT_ID}.log"

echo "[$(date '+%H:%M:%S')] 启动客户端 ${CLIENT_ID}"
echo "日志文件: ${LOG_FILE}"
mkdir -p ./logs

# 构建Python命令参数
PYTHON_ARGS=(
    --tokenizer "$TOKENIZER_PATH"
    --dataset-path "$DATASET_PATH"
    --decoder-host "$DECODER_HOST"
    --decoder-port "$DECODER_PORT"
    --num-prompts "$NUM_PROMPTS"
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

echo "[$(date '+%H:%M:%S')] 客户端 ${CLIENT_ID} 执行完成!"
echo "日志文件: ${LOG_FILE}"

#!/bin/bash
set -xe

# 配置参数
MODEL="/home/brandonye/CodeSpace/model/Qwen3-0.6B"
PREFILL_PORT=8100
DECODE_PORT=8200
PROXY_PORT=8192


# 清理函数
cleanup() {
  echo "Cleaning up..."
  pkill -f "vllm serve" || true
  pkill -f "toy_proxy_server" || true
  # pkill -f "toy_proxy_server" || true
  sleep 2
}

# 等待服务器启动
wait_for_server() {
  local port=$1
  echo "Waiting for server on port $port..."
  
  for i in {1..60}; do
    if curl -s localhost:${port}/health > /dev/null 2>&1; then
      echo "Server on port $port is ready!"
      return 0
    fi
    sleep 2
  done
  
  echo "Server on port $port failed to start"
  return 1
}

# 设置信号处理和清理
trap cleanup EXIT INT TERM

# 清理之前的实例
cleanup

# 清理日志
: > prefill_dp.log && : > decode_dp.log

echo "================================"
echo "🧪 启动DP协调器KV缓存长度统计功能集成测试"
echo "📊 Starting Data Parallel Prefill-Decode Test with KV Cache Length Statistics"
echo "🤖 模型: $MODEL"
echo "🔧 Prefill实例: 2 GPUs (0,1) - 数据并行, 端口 $PREFILL_PORT"
echo "🔧 Decode实例: 2 GPUs (0,1) - 数据并行, 端口 $DECODE_PORT"
echo "🎯 测试重点: 验证KV缓存长度统计信息从DP引擎到负载均衡器的完整数据流"
echo "📋 监控目标: lb_engines_tokens中的KV缓存长度动态变化"
echo "================================"

# 启动 Prefill 实例 (使用两张 GPU 进行数据并行)
echo "🚀 启动Prefill实例 - GPU 0,1数据并行, 端口 $PREFILL_PORT"
echo "📊 启用详细日志记录以监控KV缓存长度统计信息..."
echo "📝 日志文件: prefill_dp.log"
CUDA_VISIBLE_DEVICES=0,1 \
VLLM_LOGGING_LEVEL=INFO \
VLLM_NIXL_SIDE_CHANNEL_PORT=5559 \
UCX_TLS=tcp,cuda_copy,cuda_ipc,sm,shm \
UCX_NET_DEVICES=lo \
UCX_MEMTYPE_CACHE=n \
UCX_RNDV_SCHEME=put_zcopy \
UCX_IB_ENABLE=n \
vllm serve $MODEL \
--port $PREFILL_PORT \
--enforce-eager \
--no-enable-prefix-caching \
--disable-log-requests \
--gpu-memory-utilization 0.3 \
--max-model-len 2048 \
--tensor-parallel-size 1 \
--data-parallel-size 2 \
--max-num-seqs 8 \
--max-num-batched-tokens 64 \
--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
> prefill_dp.log 2>&1 &

# 等待 Prefill 实例启动
echo "⏳ 等待Prefill实例启动..."
wait_for_server $PREFILL_PORT
echo "✅ Prefill实例已就绪!"

# 启动 Decode 实例 (使用两张 GPU 进行数据并行)
echo "🚀 启动Decode实例 - GPU 0,1数据并行, 端口 $DECODE_PORT"
echo "📊 启用详细日志记录以监控KV缓存长度统计信息..."
echo "📝 日志文件: decode_dp.log"
echo "🎯 重点监控: DP协调器收集的KV缓存长度统计信息"
CUDA_VISIBLE_DEVICES=0,1 \
VLLM_LOGGING_LEVEL=INFO \
VLLM_NIXL_SIDE_CHANNEL_PORT=5659 \
UCX_TLS=tcp,cuda_copy,cuda_ipc,sm,shm \
UCX_NET_DEVICES=lo \
UCX_MEMTYPE_CACHE=n \
UCX_RNDV_SCHEME=put_zcopy \
UCX_IB_ENABLE=n \
vllm serve $MODEL \
--port $DECODE_PORT \
--enforce-eager \
--no-enable-prefix-caching \
--disable-log-requests \
--gpu-memory-utilization 0.3 \
--max-model-len 2048 \
--tensor-parallel-size 1 \
--data-parallel-size 2 \
--max-num-seqs 8 \
--max-num-batched-tokens 64 \
--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
> decode_dp.log 2>&1 &

# 等待 Decode 实例启动
echo "⏳ 等待Decode实例启动..."
wait_for_server $DECODE_PORT
echo "✅ Decode实例已就绪!"

echo ""
echo "🎉 PD分离架构部署完成!"
echo "📊 服务状态:"
echo "   - Prefill实例: http://localhost:$PREFILL_PORT (日志: prefill_dp.log)"
echo "   - Decode实例: http://localhost:$DECODE_PORT (日志: decode_dp.log)"
echo ""
echo "🔍 KV缓存统计监控要点:"
echo "   - 查找日志中包含'📊 收到DP统计'的条目"
echo "   - 观察'� KV缓存长度'的动态变化"
echo "   - 监控'� 全局总KV缓存长度'的增长过程"
echo "   - 验证lb_engines_tokens中的数据收集"
echo ""
echo "⚡ 现在可以运行测试客户端脚本进行KV缓存统计功能验证"
echo "🔄 脚本将保持运行状态，按Ctrl+C退出并清理资源"

# 让脚本不要退出，否则会自动调用clean up函数
wait
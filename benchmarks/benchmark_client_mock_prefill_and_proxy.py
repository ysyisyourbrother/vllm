#!/usr/bin/env python3
"""
多进程PD分离Benchmark脚本

该脚本支持多进程并发测试，每个进程使用独立的CPU核心处理请求。
主要功能：
1. 支持多进程并发，每个进程读取不同的数据段
2. 加载mooncake数据集，按顺序分配给不同进程
3. 复制proxy server的KV transfer参数生成和请求处理
4. 直接发送请求到decode instance，绕过proxy server
"""

import argparse
import asyncio
import time
import multiprocessing as mp
from typing import Optional, List
from dataclasses import dataclass
from transformers import AutoTokenizer
import httpx
import numpy as np
from tqdm.asyncio import tqdm

# 导入vllm的数据集和请求处理模块
from vllm.benchmarks.benchmark_dataset import MooncakeDataset, SampleRequest
from vllm.logger import init_logger

# 确保logger名称以vllm开头，这样可以使用vLLM的日志配置
logger_name = __name__ if __name__.startswith('vllm') else 'vllm.benchmarks.benchmark_client'
logger = init_logger(logger_name)


@dataclass
class RequestFuncOutput:
    """请求输出结果"""
    success: bool = False
    latency: float = 0.0  # in milliseconds
    ttft: float = 0.0  # Time to first token in milliseconds
    itl: List[float] = None  # Inter-token latencies in milliseconds
    tpot: float = 0.0  # Time per output token (average) in milliseconds
    prompt_len: int = 0
    output_tokens: int = 0  # Number of generated chunks
    error: str = ""

    def __post_init__(self):
        if self.itl is None:
            self.itl = []


@dataclass
class PreprocessedRequest:
    """预处理后的请求数据"""
    sample_request: SampleRequest
    kv_transfer_params: dict
    truncated_prompt: str
    input_token_count: int
    num_blocks: int
    max_tokens: int


class ProxyRequestProcessor:
    """
    Proxy请求处理器 - 复制proxy server的核心逻辑
    """

    def __init__(self, decoder_host: str, decoder_port: int, tokenizer_path: str, fixed_max_output_tokens: Optional[int] = None):
        self.decoder_host = decoder_host
        self.decoder_port = decoder_port
        self.decoder_url = f'http://{decoder_host}:{decoder_port}'
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        self.req_id_counter = 0
        self.fixed_max_output_tokens = fixed_max_output_tokens  # Support for fixed max_output_tokens

        # 创建共享的 HTTP 客户端
        self.client = httpx.AsyncClient(
            timeout=None,
            limits=httpx.Limits(
                max_connections=2000,         # 最大并发连接数
                max_keepalive_connections=500 # 空闲连接池大小
            )
        )

    def generate_kv_transfer_params(self, prompt: str) -> tuple:
        """
        生成KV transfer参数 - 完全复制proxy server逻辑
        返回: (kv_transfer_params, truncated_prompt, token_count, num_blocks)
        """
        # Get token count
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        token_count = len(tokens)

        # Calculate number of blocks (token_count // 128, rounded down)
        num_blocks = token_count // 128

        # Generate remote block IDs
        remote_block_ids = list(range(1, num_blocks + 1)) if num_blocks > 0 else []

        # Truncate prompt to (num_blocks * 128) tokens
        truncated_token_length = num_blocks * 128
        if truncated_token_length > 0:
            truncated_tokens = tokens[:truncated_token_length]
            truncated_prompt = self.tokenizer.decode(truncated_tokens, skip_special_tokens=False)
        else:
            truncated_prompt = prompt

        kv_transfer_params = {
            "do_remote_prefill": True,
            "do_remote_decode": False,
            "remote_block_ids": remote_block_ids,
            "remote_engine_id": "0",
            "remote_host": self.decoder_host,
            "remote_port": self.decoder_port,
            "remote_tp_size": 1,
        }

        return kv_transfer_params, truncated_prompt, token_count, num_blocks

    def preprocess_requests(self, input_requests: List[SampleRequest]) -> List[PreprocessedRequest]:
        """
        预处理所有请求，提前生成KV transfer参数
        """
        preprocessed_requests = []

        for sample_request in input_requests:
            # 确定max_tokens值
            max_tokens = self.fixed_max_output_tokens if self.fixed_max_output_tokens is not None else sample_request.expected_output_len

            # 生成KV transfer参数
            kv_transfer_params, truncated_prompt, input_token_count, num_blocks = self.generate_kv_transfer_params(
                sample_request.prompt
            )

            preprocessed_request = PreprocessedRequest(
                sample_request=sample_request,
                kv_transfer_params=kv_transfer_params,
                truncated_prompt=truncated_prompt,
                input_token_count=input_token_count,
                num_blocks=num_blocks,
                max_tokens=max_tokens
            )

            preprocessed_requests.append(preprocessed_request)

        return preprocessed_requests

    async def close(self):
        """关闭HTTP客户端"""
        await self.client.aclose()

    async def process_request(self, preprocessed_request: PreprocessedRequest,
                             first_chunk_callback: Optional[callable] = None) -> RequestFuncOutput:
        """
        处理单个预处理的请求 - 直接发送到decode instance

        Args:
            preprocessed_request: 预处理的请求数据
            first_chunk_callback: 收到第一个chunk时的回调函数
        """
        self.req_id_counter += 1
        request_id = str(self.req_id_counter)

        start_time = time.time()

        # 打印请求构造日志
        logger.info(f"Request {request_id} constructed: input_tokens={preprocessed_request.input_token_count}, blocks={preprocessed_request.num_blocks}, max_output_tokens={preprocessed_request.max_tokens}")

        # 构造发送到decode instance的请求
        completions_data = {
            "prompt": preprocessed_request.truncated_prompt,
            "max_tokens": preprocessed_request.max_tokens,
            "temperature": 0.0,
            "repetition_penalty": 1.0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "kv_transfer_params": preprocessed_request.kv_transfer_params
        }

        # 发送请求到decode instance
        target_url = f"{self.decoder_url}/v1/completions"

        output = RequestFuncOutput()
        output.prompt_len = preprocessed_request.sample_request.prompt_len

        try:
            headers = {"Content-Type": "application/json"}
            async with self.client.stream("POST", target_url,
                                        json=completions_data, headers=headers) as response:
                    response.raise_for_status()

                    first_token_time = None
                    chunk_count = 0
                    last_chunk = None
                    second_last_chunk = None
                    first_chunk_received = False

                    async for chunk in response.aiter_bytes():
                        if chunk:
                            current_time = time.time()

                            if first_token_time is None:
                                first_token_time = current_time
                                output.ttft = (first_token_time - start_time) * 1000  # Convert to ms

                                # 收到第一个chunk时调用回调函数更新进度条
                                if not first_chunk_received and first_chunk_callback:
                                    first_chunk_callback()
                                    first_chunk_received = True

                            chunk_count += 1

                            # 记录最后两个chunk
                            second_last_chunk = last_chunk
                            last_chunk = chunk

                    end_time = time.time()
                    output.latency = (end_time - start_time) * 1000  # Convert to ms
                    output.output_tokens = chunk_count

                    # Calculate TPOT from first token onwards (excluding TTFT)
                    if chunk_count > 0 and first_token_time is not None:
                        decode_time = end_time - first_token_time  # Time from first token to end
                        output.tpot = (decode_time / chunk_count) * 1000 if chunk_count > 0 else 0.0  # Convert to ms
                    else:
                        output.tpot = 0.0

                    output.success = True

                    # Log request completion with chunk count and second last chunk (times in ms)
                    log_msg = f"\nRequest {request_id} completed: {chunk_count} chunks received, " \
                             f"latency: {output.latency:.3f}ms, TPOT: {output.tpot:.3f}ms, second_last_chunk: {second_last_chunk}\n"

                    logger.info(log_msg)

        except Exception as e:
            output.success = False
            output.error = str(e)
            logger.info(f"Request {request_id} failed: {str(e)}")

        return output


async def run_benchmark(processor: ProxyRequestProcessor,
                       input_requests: List[SampleRequest],
                       request_rate: float = float('inf')) -> dict:
    """
    运行benchmark测试
    """
    # 预处理所有请求，提前生成KV transfer参数
    logger.info(f"Preprocessing {len(input_requests)} requests...")
    preprocessed_requests = processor.preprocess_requests(input_requests)
    logger.info(f"Preprocessing completed.")

    pbar = tqdm(total=len(preprocessed_requests), desc="Processing requests")

    async def process_with_progress(preprocessed_request):
        """处理请求并在收到第一个chunk时更新进度条"""
        def update_progress():
            """进度条更新回调函数"""
            pbar.update(1)

        result = await processor.process_request(preprocessed_request, first_chunk_callback=update_progress)
        return result

    # 控制请求速率
    if request_rate != float('inf'):
        interval = 1.0 / request_rate
        tasks = []
        start_time = time.time()

        for i, preprocessed_request in enumerate(preprocessed_requests):
            target_time = start_time + i * interval
            current_time = time.time()

            if current_time < target_time:
                await asyncio.sleep(target_time - current_time)

            task = asyncio.create_task(process_with_progress(preprocessed_request))
            tasks.append(task)
    else:
        tasks = [asyncio.create_task(process_with_progress(preprocessed_request))
                for preprocessed_request in preprocessed_requests]

    outputs = await asyncio.gather(*tasks)
    pbar.close()

    # 关闭HTTP客户端
    await processor.close()

    # 统计结果
    successful_outputs = [o for o in outputs if o.success]
    failed_outputs = [o for o in outputs if not o.success]

    if successful_outputs:
        latencies = [o.latency for o in successful_outputs]
        tpots = [o.tpot for o in successful_outputs if o.tpot > 0]
        output_chunks = [o.output_tokens for o in successful_outputs]

        # 计算统计指标，参考benchmark_serving.py
        results = {
            "total_requests": len(input_requests),
            "successful_requests": len(successful_outputs),
            "failed_requests": len(failed_outputs),
            "success_rate": len(successful_outputs) / len(input_requests),
            "avg_latency": np.mean(latencies),
            "p50_latency": np.percentile(latencies, 50),
            "p95_latency": np.percentile(latencies, 95),
            "p99_latency": np.percentile(latencies, 99),
            "avg_tpot": np.mean(tpots) if tpots else 0.0,
            "p50_tpot": np.percentile(tpots, 50) if tpots else 0.0,
            "p95_tpot": np.percentile(tpots, 95) if tpots else 0.0,
            "total_output_chunks": sum(output_chunks),
            "avg_output_chunks": np.mean(output_chunks) if output_chunks else 0.0,
        }
    else:
        results = {
            "total_requests": len(input_requests),
            "successful_requests": 0,
            "failed_requests": len(failed_outputs),
            "success_rate": 0.0,
            "avg_latency": 0.0,
            "p50_latency": 0.0,
            "p95_latency": 0.0,
            "p99_latency": 0.0,
            "avg_tpot": 0.0,
            "p50_tpot": 0.0,
            "p95_tpot": 0.0,
            "total_output_chunks": 0,
            "avg_output_chunks": 0.0,
        }

    return results


def create_argument_parser():
    """创建参数解析器"""
    parser = argparse.ArgumentParser(description="Multi-process PD separation benchmark")

    # Decode instance配置
    parser.add_argument("--decoder-host", type=str, default="localhost",
                       help="Decode instance host")
    parser.add_argument("--decoder-port", type=int, default=8002,
                       help="Decode instance port")
    parser.add_argument("--tokenizer", type=str, required=True,
                       help="Tokenizer path")

    # 数据集配置
    parser.add_argument("--dataset-name", type=str, default="mooncake",
                       help="Dataset name (only mooncake supported)")
    parser.add_argument("--dataset-path", type=str, required=True,
                       help="Dataset file path")
    parser.add_argument("--num-prompts", type=int, default=100,
                       help="Number of prompts per client")

    # 多进程配置
    parser.add_argument("--num-clients", type=int, required=True,
                       help="Number of client processes")

    # Benchmark配置
    parser.add_argument("--request-rate", type=float, default=float('inf'),
                       help="Request rate (RPS)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--fixed-max-output-tokens", type=int, default=None,
                       help="Fixed max output tokens for all requests (overrides dataset values)")

    return parser


def load_dataset_segment(dataset_path: str, tokenizer_path: str,
                        client_id: int, num_clients: int, num_prompts: int, seed: int) -> List[SampleRequest]:
    """
    为指定的客户端加载数据集的轮询分配段

    轮询分配逻辑：假设10个客户端，数据集前1-10条分别给客户端0-9，
    第11-20条又分别给客户端0-9，以此类推

    Args:
        dataset_path: 数据集路径
        tokenizer_path: tokenizer路径
        client_id: 客户端ID (0-based)
        num_clients: 总客户端数
        num_prompts: 每个客户端的请求数
        seed: 随机种子

    Returns:
        该客户端应处理的请求列表
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    dataset = MooncakeDataset(random_seed=seed, dataset_path=dataset_path)

    # 轮询分配：每个客户端获取间隔为num_clients的数据
    input_requests = []
    for i in range(num_prompts):
        # 计算该客户端第i个请求对应的数据集索引
        data_index = (i * num_clients + client_id) % len(dataset.data)
        entry = dataset.data[data_index]

        # 生成合成的prompt
        input_length = entry['input_length']
        output_length = entry['output_length']
        timestamp = entry['timestamp']

        vocab_size = tokenizer.vocab_size
        token_ids = [(timestamp + data_index + j) % vocab_size for j in range(input_length)]
        prompt = tokenizer.decode(token_ids)

        input_requests.append(
            SampleRequest(
                prompt=prompt,
                prompt_len=input_length,
                expected_output_len=output_length,
            )
        )

    return input_requests


async def run_single_client(client_id: int, args, sync_barrier) -> dict:
    """
    运行单个客户端进程

    Args:
        client_id: 客户端ID
        args: 命令行参数
        sync_barrier: 同步屏障，用于所有进程同时开始

    Returns:
        该客户端的benchmark结果
    """
    # 加载该客户端的数据段
    input_requests = load_dataset_segment(
        args.dataset_path, args.tokenizer, client_id,
        args.num_clients, args.num_prompts, args.seed
    )

    # 创建processor
    processor = ProxyRequestProcessor(
        decoder_host=args.decoder_host,
        decoder_port=args.decoder_port,
        tokenizer_path=args.tokenizer,
        fixed_max_output_tokens=args.fixed_max_output_tokens
    )

    logger.info(f"Client {client_id}: Loaded {len(input_requests)} requests, waiting for sync...")

    # 等待所有进程准备完毕
    sync_barrier.wait()

    logger.info(f"Client {client_id}: Starting benchmark...")

    # 运行benchmark
    results = await run_benchmark(processor, input_requests, args.request_rate)
    results['client_id'] = client_id

    return results


def run_client_process(client_id: int, args, sync_barrier, results_queue):
    """
    客户端进程的入口函数
    """
    try:
        # 运行异步客户端
        results = asyncio.run(run_single_client(client_id, args, sync_barrier))
        results_queue.put(results)
    except Exception as e:
        logger.error(f"Client {client_id} failed: {str(e)}")
        results_queue.put({'client_id': client_id, 'error': str(e)})


def aggregate_results(all_results: List[dict]) -> dict:
    """
    聚合所有客户端的结果
    """
    # 过滤出成功的结果
    successful_results = [r for r in all_results if 'error' not in r]
    failed_results = [r for r in all_results if 'error' in r]

    if not successful_results:
        logger.error("All clients failed!")
        return {}

    # 聚合统计数据
    total_requests = sum(r['total_requests'] for r in successful_results)
    successful_requests = sum(r['successful_requests'] for r in successful_results)
    failed_requests = sum(r['failed_requests'] for r in successful_results) + len(failed_results)

    # 收集所有延迟数据进行重新计算
    all_latencies = []
    all_tpots = []
    total_output_chunks = 0

    for r in successful_results:
        # 这里我们只能使用已聚合的统计数据，无法获取原始数据
        # 所以使用加权平均来近似
        if r['successful_requests'] > 0:
            all_latencies.extend([r['avg_latency']] * r['successful_requests'])
            if r['avg_tpot'] > 0:
                all_tpots.extend([r['avg_tpot']] * r['successful_requests'])
            total_output_chunks += r['total_output_chunks']

    if all_latencies:
        aggregated_results = {
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "success_rate": successful_requests / total_requests if total_requests > 0 else 0.0,
            "avg_latency": np.mean(all_latencies),
            "p50_latency": np.percentile(all_latencies, 50),
            "p95_latency": np.percentile(all_latencies, 95),
            "p99_latency": np.percentile(all_latencies, 99),
            "avg_tpot": np.mean(all_tpots) if all_tpots else 0.0,
            "p50_tpot": np.percentile(all_tpots, 50) if all_tpots else 0.0,
            "p95_tpot": np.percentile(all_tpots, 95) if all_tpots else 0.0,
            "total_output_chunks": total_output_chunks,
            "avg_output_chunks": total_output_chunks / successful_requests if successful_requests > 0 else 0.0,
        }
    else:
        aggregated_results = {
            "total_requests": total_requests,
            "successful_requests": 0,
            "failed_requests": failed_requests,
            "success_rate": 0.0,
            "avg_latency": 0.0,
            "p50_latency": 0.0,
            "p95_latency": 0.0,
            "p99_latency": 0.0,
            "avg_tpot": 0.0,
            "p50_tpot": 0.0,
            "p95_tpot": 0.0,
            "total_output_chunks": 0,
            "avg_output_chunks": 0.0,
        }

    return aggregated_results


def main():
    """主函数"""
    parser = create_argument_parser()
    args = parser.parse_args()

    if args.dataset_name != "mooncake":
        raise ValueError("Only mooncake dataset is supported")

    # 多进程模式
    run_multi_process_mode(args)


def run_multi_process_mode(args):
    """多进程模式"""
    logger.info(f"Starting {args.num_clients} client processes...")

    # 创建同步屏障和结果队列
    sync_barrier = mp.Barrier(args.num_clients)
    results_queue = mp.Queue()

    # 启动所有客户端进程
    processes = []
    for client_id in range(args.num_clients):
        p = mp.Process(
            target=run_client_process,
            args=(client_id, args, sync_barrier, results_queue)
        )
        p.start()
        processes.append(p)

    # 等待所有进程完成并收集结果
    all_results = []
    for _ in range(args.num_clients):
        result = results_queue.get()
        all_results.append(result)

    # 等待所有进程结束
    for p in processes:
        p.join()

    # 聚合并打印结果
    aggregated_results = aggregate_results(all_results)
    if aggregated_results:
        logger.info(f"\nAggregated Results from {args.num_clients} clients:")
        print_results(aggregated_results)

        # 打印每个客户端的详细结果
        logger.info(f"\nPer-client Results:")
        for result in all_results:
            if 'error' in result:
                logger.info(f"Client {result['client_id']}: FAILED - {result['error']}")
            else:
                logger.info(f"Client {result['client_id']}: {result['successful_requests']}/{result['total_requests']} successful, "
                           f"avg_latency: {result['avg_latency']:.3f}ms")


def print_results(results: dict):
    """打印结果"""
    logger.info(f"Total requests: {results['total_requests']}")
    logger.info(f"Successful: {results['successful_requests']}")
    logger.info(f"Failed: {results['failed_requests']}")
    logger.info(f"Success rate: {results['success_rate']:.2%}")
    logger.info(f"Average latency: {results['avg_latency']:.3f}ms")
    logger.info(f"P50 latency: {results['p50_latency']:.3f}ms")
    logger.info(f"P95 latency: {results['p95_latency']:.3f}ms")
    logger.info(f"P99 latency: {results['p99_latency']:.3f}ms")
    logger.info(f"Average TPOT: {results['avg_tpot']:.3f}ms")
    logger.info(f"P50 TPOT: {results['p50_tpot']:.3f}ms")
    logger.info(f"P95 TPOT: {results['p95_tpot']:.3f}ms")
    logger.info(f"Total output chunks: {results['total_output_chunks']}")
    logger.info(f"Average output chunks per request: {results['avg_output_chunks']:.1f}")


if __name__ == "__main__":
    main()

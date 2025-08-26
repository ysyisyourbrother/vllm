# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import itertools
import os
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from vllm.logger import init_logger
import logging

logger = init_logger(__name__)
# 确保代理服务器的日志级别为INFO
logger.setLevel(logging.INFO)

# 添加控制台处理器确保日志输出
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup and shutdown events.
    """
    # Startup: Initialize client pools for prefiller and decoder services
    app.state.prefill_clients = []
    app.state.decode_clients = []

    # Create prefill clients
    for i, (host, port) in enumerate(global_args.prefiller_instances):
        prefiller_base_url = f'http://{host}:{port}/v1'
        app.state.prefill_clients.append({
            'client': httpx.AsyncClient(timeout=None, 
                              base_url=prefiller_base_url,
                              limits=httpx.Limits(
                              max_connections=500,      # 增加最大连接数
                              max_keepalive_connections=100  # 增加保持活跃连接
                              )),
            'host': host,
            'port': port,
            'id': i
        })

    # Create decode clients
    for i, (host, port) in enumerate(global_args.decoder_instances):
        decoder_base_url = f'http://{host}:{port}/v1'
        app.state.decode_clients.append({
            'client': httpx.AsyncClient(timeout=None, 
                              base_url=decoder_base_url,
                              limits=httpx.Limits(
                              max_connections=500,      # 增加最大连接数
                              max_keepalive_connections=100  # 增加保持活跃连接
                              )),
            'host': host,
            'port': port,
            'id': i
        })

    # Initialize round-robin iterators
    app.state.prefill_iterator = itertools.cycle(
        range(len(app.state.prefill_clients)))
    app.state.decode_iterator = itertools.cycle(
        range(len(app.state.decode_clients)))

    print(f"Initialized {len(app.state.prefill_clients)} prefill clients "
          f"and {len(app.state.decode_clients)} decode clients.")

    yield

    # Shutdown: Close all clients
    for client_info in app.state.prefill_clients:
        await client_info['client'].aclose()

    for client_info in app.state.decode_clients:
        await client_info['client'].aclose()


# Update FastAPI app initialization to use lifespan
app = FastAPI(lifespan=lifespan)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="localhost")

    # For prefiller instances
    parser.add_argument("--prefiller-hosts",
                        "--prefiller-host",
                        type=str,
                        nargs="+",
                        default=["localhost"])
    parser.add_argument("--prefiller-ports",
                        "--prefiller-port",
                        type=int,
                        nargs="+",
                        default=[8100])

    # For decoder instances
    parser.add_argument("--decoder-hosts",
                        "--decoder-host",
                        type=str,
                        nargs="+",
                        default=["localhost"])
    parser.add_argument("--decoder-ports",
                        "--decoder-port",
                        type=int,
                        nargs="+",
                        default=[8200])

    args = parser.parse_args()

    # Validate and pair hosts with ports
    if len(args.prefiller_hosts) != len(args.prefiller_ports):
        raise ValueError(
            "Number of prefiller hosts must match number of prefiller ports")

    if len(args.decoder_hosts) != len(args.decoder_ports):
        raise ValueError(
            "Number of decoder hosts must match number of decoder ports")

    # Create tuples of (host, port) for each service type
    args.prefiller_instances = list(
        zip(args.prefiller_hosts, args.prefiller_ports))
    args.decoder_instances = list(zip(args.decoder_hosts, args.decoder_ports))

    return args


def get_next_client(app, service_type: str):
    """
    Get the next client in round-robin fashion.

    Args:
        app: The FastAPI app instance
        service_type: Either 'prefill' or 'decode'

    Returns:
        The next client to use
    """
    if service_type == 'prefill':
        client_idx = next(app.state.prefill_iterator)
        return app.state.prefill_clients[client_idx]
    elif service_type == 'decode':
        client_idx = next(app.state.decode_iterator)
        return app.state.decode_clients[client_idx]
    else:
        raise ValueError(f"Unknown service type: {service_type}")


async def send_request_to_service(client_info: dict, endpoint: str,
                                  req_data: dict, request_id: str, original_headers: dict = None):
    """
    Send a request to a service using a client from the pool.
    """
    req_data = req_data.copy()
    req_data['kv_transfer_params'] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None
    }
    # Prefill 阶段关闭流式输出
    req_data["stream"] = False
    req_data["max_tokens"] = 1
    if "stream_options" in req_data:
        del req_data["stream_options"]
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id
    }

    # 传递原始请求的benchmark头信息
    if original_headers:
        for key in ["X-Benchmark-Input-Length", "X-Benchmark-Output-Length"]:
            if key in original_headers:
                headers[key] = original_headers[key]

    response = await client_info['client'].post(endpoint,
                                                json=req_data,
                                                headers=headers)
    response.raise_for_status()

    return response


async def stream_service_response(client_info: dict, endpoint: str,
                                  req_data: dict, request_id: str, original_headers: dict = None):
    """
    Asynchronously stream response from a service using a client from the pool.
    """
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id
    }

    # 传递原始请求的benchmark头信息
    if original_headers:
        for key in ["X-Benchmark-Input-Length", "X-Benchmark-Output-Length"]:
            if key in original_headers:
                headers[key] = original_headers[key]

    # Decode 阶段，req_data["stream"] = True: 使用了原始的req_data
    async with client_info['client'].stream("POST",
                                            endpoint,
                                            json=req_data,
                                            headers=headers) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            yield chunk


async def _handle_completions(api: str, request: Request):
    try:
        req_data = await request.json()
        request_id = str(uuid.uuid4())

        # 从请求头中获取准确的input_length和expected_output_length
        input_length = None
        expected_output_length = None

        # 尝试从benchmark请求头中获取准确数值
        if "X-Benchmark-Input-Length" in request.headers:
            try:
                input_length = int(request.headers["X-Benchmark-Input-Length"])
            except (ValueError, TypeError):
                pass

        if "X-Benchmark-Output-Length" in request.headers:
            try:
                expected_output_length = int(request.headers["X-Benchmark-Output-Length"])
            except (ValueError, TypeError):
                pass

        # 【节点1】代理服务器接收请求
        if os.getenv('VLLM_REQUEST_LOG_DEBUG', 'false').lower() == 'true':
            log_msg = f"【{request_id}，代理服务器接收请求】"
            if input_length is not None:
                log_msg += f" input_length={input_length}"
            if expected_output_length is not None:
                log_msg += f", expected_output_length={expected_output_length}"
            logger.info(log_msg)

        # Get the next prefill client in round-robin fashion
        prefill_client_info = get_next_client(request.app, 'prefill')

        # Send request to prefill service
        response = await send_request_to_service(prefill_client_info, api,
                                                 req_data, request_id, request.headers)

        # Extract the needed fields
        response_json = response.json()
        kv_transfer_params = response_json.get('kv_transfer_params', {})
        if kv_transfer_params:
            req_data["kv_transfer_params"] = kv_transfer_params

        # Get the next decode client in round-robin fashion
        decode_client_info = get_next_client(request.app, 'decode')

        logger.debug("Using %s %s", prefill_client_info, decode_client_info)

        # Stream response from decode service
        async def generate_stream():
            async for chunk in stream_service_response(decode_client_info,
                                                       api,
                                                       req_data,
                                                       request_id=request_id,
                                                       original_headers=request.headers):
                yield chunk

        # 【节点10】请求返回用户
        if os.getenv('VLLM_REQUEST_LOG_DEBUG', 'false').lower() == 'true':
            log_msg = f"【{request_id}，请求返回用户】"
            if input_length is not None:
                log_msg += f" input_length={input_length}"
            if expected_output_length is not None:
                log_msg += f", expected_output_length={expected_output_length}"
            logger.info(log_msg)

        return StreamingResponse(generate_stream(),
                                media_type="application/json")

    except Exception as e:
        import sys
        import traceback
        exc_info = sys.exc_info()
        print("Error occurred in disagg prefill proxy server"
              f" - {api} endpoint")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))
        raise


@app.post("/v1/completions")
async def handle_completions(request: Request):
    return await _handle_completions("/completions", request)


@app.post("/v1/chat/completions")
async def handle_chat_completions(request: Request):
    return await _handle_completions("/chat/completions", request)


@app.get("/healthcheck")
async def healthcheck():
    """Simple endpoint to check if the server is running."""
    return {
        "status": "ok",
        "prefill_instances": len(app.state.prefill_clients),
        "decode_instances": len(app.state.decode_clients)
    }


if __name__ == '__main__':
    global global_args
    global_args = parse_args()

    import uvicorn
    uvicorn.run(app, host=global_args.host, port=global_args.port)

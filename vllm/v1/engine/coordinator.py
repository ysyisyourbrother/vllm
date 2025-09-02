# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import multiprocessing
import time
import weakref
from typing import Optional

import msgspec.msgpack
import zmq

from vllm.config import ParallelConfig
from vllm.logger import init_logger
from vllm.utils import get_mp_context, make_zmq_socket
from vllm.v1.engine import EngineCoreOutputs, EngineCoreRequestType
from vllm.v1.serial_utils import MsgpackDecoder
from vllm.v1.utils import get_engine_client_zmq_addr, shutdown

logger = init_logger(__name__)


class DPCoordinator:
    """用于数据并行部署(DP>1)的协调器进程

    Coordinator process used for data-parallel deployments (DP>1).

    在多个DP引擎rank进程和一个或多个前端API服务器进程之间进行中介。
    Intermediates between multiple DP engine rank processes and one or more
    front-end API server processes.

    * 从每个DP引擎收集统计信息（包括等待和运行队列长度，以及KV缓存长度），
      并将这些信息发布给所有前端，用于负载均衡决策。
      Collects stats from each DP engine (including waiting and running
      queue lengths, and KV cache lengths), and publishes these to all
      front-ends for use in load-balancing decisions.

    * 跟踪当前DP"请求波次"号和引擎的运行状态。这些信息从DP rank 0引擎接收，
      并与当前负载统计一起发布给前端进程。
      Keeps track of the current DP "request wave" number and running state
      of the engines. This is received from the DP rank 0 engine and published
      to the front-end processes along with the current load stats.

      引擎在全局运行/暂停状态之间交替。全局"请求波次"号是工作进程集体从运行状态
      转移到暂停状态次数的计数。这种转换通过DPEngineCoreProc._has_global_unfinished_reqs
      方法中执行的all-reduce操作进行同步。
      The engines alternate between a global running/paused state. The global
      "request wave" number is a count of the number of times that the workers
      collectively move from a running state to a paused state. This transition
      is synchronized via the all-reduce operation performed in the
      DPEngineCoreProc._has_global_unfinished_reqs method.

    * Broadcasts the START_DP_WAVE message to engines to move them from paused
      to running state when one engine receives a new request. This can happen
      in two cases:
      1) A front-end sending a new request while the engines are paused will
         concurrently notify the coordinator.
      2) An engine receiving a request for a stale request wave while in paused
         state will notify the coordinator.

    Engines will move into running state when receiving a new request or
    START_DP_WAVE message.

    Note that when deployed in External LB mode, no stats will be published by
    the engines and thus updates will only be sent to front-ends when the
    request wave / running state changes.
    """

    def __init__(self, parallel_config: ParallelConfig):

        dp_size = parallel_config.data_parallel_size
        assert dp_size > 1, "Coordinator only used for data parallel"

        host = parallel_config.data_parallel_master_ip
        external_lb = parallel_config.data_parallel_external_lb
        hybrid_lb = parallel_config.data_parallel_hybrid_lb

        # Assume coordinator is colocated with front-end procs when not in
        # either external or hybrid DP LB mode.
        front_publish_address = get_engine_client_zmq_addr(
            local_only=not external_lb and not hybrid_lb, host=host)

        local_only_eng = dp_size == parallel_config.data_parallel_size_local
        back_publish_address = get_engine_client_zmq_addr(local_only_eng, host)
        back_output_address = get_engine_client_zmq_addr(local_only_eng, host)

        # When in external LB mode, load stats aren't published, only changes
        # to request wave / running state, so we don't need to rate-limit the
        # updates to the front-end proc(s).

        # Brandon 修改统计更新频率为10ms以提高DPLBAsyncMPClient监控的实时性
        # Modified stats update frequency to 10ms for better real-time monitoring in DPLBAsyncMPClient
        min_stats_update_interval_ms = 0 if external_lb else 10

        context = get_mp_context()
        self.proc: multiprocessing.Process = context.Process(
            target=CoordinatorProc.run_coordinator,
            name="VLLM_DP_Coordinator",
            kwargs={
                "engine_count": parallel_config.data_parallel_size,
                "front_publish_address": front_publish_address,
                "back_output_address": back_output_address,
                "back_publish_address": back_publish_address,
                "min_stats_update_interval_ms": min_stats_update_interval_ms,
            },
            daemon=True)
        self.proc.start()

        self.stats_publish_address = front_publish_address
        self.coord_in_address = back_publish_address
        self.coord_out_address = back_output_address
        self._finalizer = weakref.finalize(self, shutdown, [self.proc])

    def get_stats_publish_address(self) -> str:
        return self.stats_publish_address

    def get_engine_socket_addresses(self) -> tuple[str, str]:
        """Returns tuple of ZMQ input address, output address."""
        return self.coord_in_address, self.coord_out_address

    def close(self):
        self._finalizer()


class EngineState:
    """存储单个DP引擎的状态信息

    Stores state information for a single DP engine.
    """

    def __init__(self):
        # 请求计数 [等待中, 运行中] - 用于现有的负载均衡逻辑
        # Request counts [waiting, running] - for existing load balancing logic
        self.request_counts = [0, 0]  # [waiting, running]

        # KV缓存长度 - 该引擎所有请求的总KV缓存长度
        # KV cache length - total KV cache length of all requests in this engine
        self.kv_cache_length = 0


class CoordinatorProc:

    def __init__(self,
                 engine_count: int,
                 min_stats_update_interval_ms: int = 100):

        self.ctx = zmq.Context()

        self.engines = [EngineState() for _ in range(engine_count)]

        self.stats_update_interval_ms = min_stats_update_interval_ms

        self.current_wave = 0
        self.engines_running = False
        self.stats_changed = False

    @staticmethod
    def run_coordinator(
        engine_count: int,
        front_publish_address: str,
        back_output_address: str,
        back_publish_address: str,
        min_stats_update_interval_ms: int = 100,
    ):
        coordinator = CoordinatorProc(
            engine_count=engine_count,
            min_stats_update_interval_ms=min_stats_update_interval_ms)
        try:
            coordinator.process_input_socket(
                front_publish_address,
                back_output_address,
                back_publish_address,
            )
        except KeyboardInterrupt:
            logger.info("DP Coordinator process exiting")

    def process_input_socket(self, front_publish_address: str,
                             back_output_address: str,
                             back_publish_address: str):

        decoder = MsgpackDecoder(EngineCoreOutputs)

        with make_zmq_socket(
                path=front_publish_address,  # IPC
                ctx=self.ctx,
                socket_type=zmq.XPUB,
                bind=True,
        ) as publish_front, make_zmq_socket(
                path=back_output_address,  # IPC or TCP
                ctx=self.ctx,
                socket_type=zmq.PULL,
                bind=True,
        ) as output_back, make_zmq_socket(
                path=back_publish_address,  # IPC or TCP
                ctx=self.ctx,
                socket_type=zmq.XPUB,
                bind=True,
        ) as publish_back:

            poller = zmq.Poller()
            poller.register(publish_front, zmq.POLLIN)
            poller.register(output_back, zmq.POLLIN)
            last_publish_time = 0
            while True:
                elapsed = int(time.time() * 1000) - last_publish_time
                # Send at stats_update_interval_ms interval if the stats have
                # changed, or otherwise every 4 seconds.
                wait_for = (self.stats_update_interval_ms
                            if self.stats_changed else 4000)
                events = poller.poll(timeout=max(0, wait_for - elapsed))
                if not events:
                    # Poller timeout - 发布当前统计信息到前端进程
                    # Poller timeout - publish current stats to front-ends
                    engine_req_counts_list = self._get_engine_counts()
                    engine_kv_cache_lengths = self._get_engine_kv_cache_lengths()
                    total_kv_cache_length = sum(engine_kv_cache_lengths)

                    # 发布的消息格式：(请求计数列表, 当前波次, 引擎运行状态, KV缓存长度列表, 总KV缓存长度)
                    # Published message format: (request_counts_list, current_wave, engines_running, kv_cache_lengths, total_kv_cache_length)
                    to_publish = (engine_req_counts_list, self.current_wave,
                                  self.engines_running, engine_kv_cache_lengths,
                                  total_kv_cache_length)
                    publish_front.send(msgspec.msgpack.encode(to_publish))
                    last_publish_time = int(time.time() * 1000)
                    self.stats_changed = False
                    continue

                events = dict(events)

                if publish_front in events:
                    buffer = publish_front.recv()
                    if buffer in (b'\x01', b'\x00'):
                        # Ignore subscription messages.
                        continue

                    decoded = msgspec.msgpack.decode(buffer)
                    if isinstance(decoded, (list, tuple)) and len(
                            decoded) == 2 and decoded[0] == "SCALE_ELASTIC_EP":
                        # Handle scale up notification
                        new_engine_count = decoded[1]
                        current_count = len(self.engines)
                        if new_engine_count > current_count:
                            for _ in range(new_engine_count - current_count):
                                self.engines.append(EngineState())
                            # NOTE(yongji): handle the case
                            # where newly started engines have current_wave = 0
                            # if existing engines just finished a wave
                            # and engine_running isn't updated yet at
                            # CoordinatorProc requests routed to newly started
                            # engines may not wake up existing engines, as long
                            # as 0 < request.wave < existing engines'
                            # current_wave
                            # we note that 0 is the wave number for the new
                            # engine
                            self.engines_running = False
                            logger.info(
                                "DPCoordinator scaled up from %s to %s "
                                "engines", current_count, new_engine_count)
                        else:
                            self.engines = self.engines[:new_engine_count]
                            logger.info(
                                "DPCoordinator scaled down from %s to %s "
                                "engines", current_count, new_engine_count)
                        continue  # Skip normal engine notification processing

                    # We received a message on the front-end XPUB socket,
                    # from an API server sending a new request while the
                    # engines are paused, so that we can wake the other
                    # engines.
                    engine_to_exclude, wave = decoded
                    if not self.engines_running:
                        if wave < self.current_wave:
                            # If the wave number is stale, ensure the message
                            # is handled by all the engines.
                            engine_to_exclude = None

                        self.engines_running = True
                        self.stats_changed = True
                        self._send_start_wave(publish_back, self.current_wave,
                                              engine_to_exclude)

                if output_back in events:
                    # We received a message from one of the engines.

                    buffer = output_back.recv()
                    outputs: EngineCoreOutputs = decoder.decode(buffer)

                    assert not outputs.outputs
                    assert outputs.utility_output is None

                    eng_index = outputs.engine_index
                    scheduler_stats = outputs.scheduler_stats

                    if scheduler_stats:
                        # 1. 更新请求负载统计信息 - 更新本地状态
                        # 1. Updated request load stats - update our local state
                        engine_state = self.engines[eng_index]
                        stats = engine_state.request_counts
                        old_kv_length = engine_state.kv_cache_length

                        stats[0] = scheduler_stats.num_waiting_reqs
                        stats[1] = scheduler_stats.num_running_reqs

                        # 2. 更新KV缓存长度统计信息
                        # 2. Update KV cache length statistics
                        engine_state.kv_cache_length = scheduler_stats.total_kv_cache_length

                        self.stats_changed = True

                    if (wave := outputs.wave_complete) is not None:
                        # 2. Notification from rank 0 engine that we've
                        # moved into the global paused state
                        # (engines_running==False).
                        if self.current_wave <= wave:
                            new_wave = wave + 1
                            logger.debug("Moving DP wave from %d to %d.",
                                         self.current_wave, new_wave)
                            self.current_wave = new_wave
                            self.engines_running = False
                            self.stats_changed = True
                    elif (wave := outputs.start_wave) is not None and (
                            wave > self.current_wave or
                        (wave == self.current_wave
                         and not self.engines_running)):
                        # 3. The engine received request for a non-current wave
                        # so we must ensure that other engines progress to the
                        # next wave (race condition handling).
                        logger.debug(
                            "Starting wave %d after notification of "
                            "stale wave request from engine.", wave)
                        self.current_wave = wave
                        self.engines_running = True
                        self.stats_changed = True
                        self._send_start_wave(publish_back, wave, eng_index)

    @staticmethod
    def _send_start_wave(socket: zmq.Socket, wave: int,
                         exclude_engine_index: Optional[int]):
        """Broadcast the START_DP_WAVE message to all the engines.
        It includes the current wave number and index of engine which
        has already received a request with this wave number and so doesn't
        require additional notification.
        """
        wave_encoded = msgspec.msgpack.encode((wave, exclude_engine_index))
        socket.send_multipart(
            (EngineCoreRequestType.START_DP_WAVE.value, wave_encoded))

    def _get_engine_counts(self) -> list[list[int]]:
        """返回每个引擎的[等待中, 运行中]请求计数列表

        Return list of [waiting, running] count lists for each engine.
        """
        return [e.request_counts for e in self.engines]

    def _get_engine_kv_cache_lengths(self) -> list[int]:
        """返回每个引擎的KV缓存长度列表

        Return list of KV cache lengths for each engine.

        Returns:
            list[int]: 每个引擎的KV缓存长度 / KV cache length for each engine
        """
        return [e.kv_cache_length for e in self.engines]

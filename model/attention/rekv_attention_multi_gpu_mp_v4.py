import torch
import torch.multiprocessing as mp
import queue

from .kv_cache_manager_v3 import ContextManager
from .position_bias_utils import load_position_bias

# NVTX
try:
    from torch.cuda import nvtx
    NVTX_AVAILABLE = True
except ImportError:
    NVTX_AVAILABLE = False
    class DummyNVTX:
        @staticmethod
        def range_push(msg):
            pass
        @staticmethod
        def range_pop():
            pass
    nvtx = DummyNVTX()

# 설정
batch_size = 1
num_heads = 32
dim_head = 128
seq_length = 8192
chunk_size = 256
exc_block_size = 256
fattn = True


def worker_gpu0(q_gpu0, k_gpu0, v_gpu0, cross_request_queue, cross_response_queue,
                result_queue, barrier_start, barrier_warmup, chunk_barriers, barrier_end):
    """
    GPU0 Worker (Stream 기반):
    각 청크마다:
    1. Stream 1: Local attention
    2. Stream 2: GPU1로부터 q 받기
    3. 동기화 후 Cross attention
    4. Barrier로 GPU1과 동기화
    """
    torch.cuda.set_device(0)
    
    # Context Manager 초기화
    position_bias = load_position_bias(
        "/root/mwnoh/ReKV/model/attention/position_bias.pkl", device="cuda:0"
    )
    context_manager = ContextManager(
        position_bias, exc_block_size=exc_block_size, fattn=fattn
    )
    context_manager.init(
        batch_size=batch_size, num_heads=num_heads, dim_head=dim_head,
        dtype=torch.float16, device='cuda:0'
    )
    
    # 데이터를 GPU0으로 이동
    q_gpu0 = q_gpu0.to('cuda:0')
    k_gpu0 = k_gpu0.to('cuda:0')
    v_gpu0 = v_gpu0.to('cuda:0')
    
    seq_len = q_gpu0.size(-2)
    num_chunks = (seq_len + chunk_size - 1) // chunk_size
    
    # Stream 생성
    stream_local = torch.cuda.Stream()
    stream_transfer = torch.cuda.Stream()
    
    # 결과 저장
    local_results = []
    local_time_events = []
    cross_time_events = []
    q_receive_time_events = []
    o_send_time_events = []
    
    # 시작 동기화
    barrier_start.wait()
    
    nvtx.range_push("GPU0_Worker")
    
    # === Warmup Phase ===
    print("GPU0: Warmup phase...")
    nvtx.range_push("GPU0_Warmup")
    
    # Warmup: Local attention
    q_warmup = q_gpu0[:, :, 0:chunk_size, :]
    k_warmup = k_gpu0[:, :, 0:chunk_size, :]
    v_warmup = v_gpu0[:, :, 0:chunk_size, :]
    with torch.cuda.stream(stream_local):
        _ = context_manager._append(q_warmup, k_warmup, v_warmup)
    
    # Warmup: Transfer (dummy receive)
    with torch.cuda.stream(stream_transfer):
        # GPU1에서 보낼 첫 번째 요청 받기
        try:
            warmup_request = cross_request_queue.get(timeout=10)
            q_warmup_from_gpu1 = warmup_request['q_chunk']
        except queue.Empty:
            print("GPU0: Warmup transfer failed")
            barrier_end.wait()
            return
    
    # Warmup: Cross attention
    _ = context_manager._append(q_warmup_from_gpu1, k_gpu0, v_gpu0)
    warmup_result = _.to('cuda:1', non_blocking=True)
    warmup_result.share_memory_()
    cross_response_queue.put({'chunk_idx': -1, 'result': warmup_result})
    
    torch.cuda.synchronize('cuda:0')
    nvtx.range_pop()
    print("GPU0: Warmup completed")
    
    # Warmup 완료 동기화
    barrier_warmup.wait()
    
    total_start = torch.cuda.Event(enable_timing=True)
    total_end = torch.cuda.Event(enable_timing=True)
    total_start.record()
    
    print(f"GPU0: Processing {num_chunks} chunks with stream-based parallelism")
    
    for chunk_idx in range(num_chunks):
        nvtx.range_push(f"GPU0_Chunk{chunk_idx}")
        
        st = chunk_idx * chunk_size
        ed = min(st + chunk_size, seq_len)
        
        # === 병렬 실행: Local attention + Transfer 받기 ===
        
        # Stream 1: Local attention
        nvtx.range_push(f"GPU0_Local_Chunk{chunk_idx}")
        local_start = torch.cuda.Event(enable_timing=True)
        local_end = torch.cuda.Event(enable_timing=True)
        
        with torch.cuda.stream(stream_local):
            local_start.record(stream_local)
            
            q_chunk = q_gpu0[:, :, st:ed, :]
            k_causal = k_gpu0[:, :, 0:ed, :]
            v_causal = v_gpu0[:, :, 0:ed, :]
            
            local_result = context_manager._append(q_chunk, k_causal, v_causal)
            local_results.append(local_result)
            
            local_end.record(stream_local)
        
        nvtx.range_pop()
        
        # Stream 2: GPU1로부터 Cross 요청 받기
        nvtx.range_push(f"GPU0_Transfer_Receive_Chunk{chunk_idx}")
        q_receive_start = torch.cuda.Event(enable_timing=True)
        q_receive_end = torch.cuda.Event(enable_timing=True)
        
        with torch.cuda.stream(stream_transfer):
            q_receive_start.record(stream_transfer)
            
            # Queue에서 요청 받기 (CPU 작업, stream과 무관)
            try:
                request = cross_request_queue.get(timeout=10)
                q_chunk_from_gpu1 = request['q_chunk']
                recv_chunk_idx = request['chunk_idx']
                assert recv_chunk_idx == chunk_idx, f"Chunk mismatch: expected {chunk_idx}, got {recv_chunk_idx}"
            except queue.Empty:
                print(f"GPU0: Timeout waiting for cross request chunk {chunk_idx}")
                break
            
            q_receive_end.record(stream_transfer)
        
        nvtx.range_pop()
        
        # === 동기화: Local + Transfer 완료 대기 ===
        torch.cuda.synchronize('cuda:0')
        
        local_time_events.append((local_start, local_end))
        q_receive_time_events.append((q_receive_start, q_receive_end))
        
        # === 순차 실행: Cross attention ===
        nvtx.range_push(f"GPU0_Cross_Chunk{chunk_idx}")
        cross_start = torch.cuda.Event(enable_timing=True)
        cross_end = torch.cuda.Event(enable_timing=True)
        cross_start.record()
        cross_result = context_manager._append(q_chunk_from_gpu1, k_gpu0, v_gpu0)
        cross_end.record()
        nvtx.range_pop()
        cross_time_events.append((cross_start, cross_end))
        
        # GPU0→GPU1 직접 전송 (비동기)
        nvtx.range_push(f"GPU0_Send_Chunk{chunk_idx}")
        o_send_start = torch.cuda.Event(enable_timing=True)
        o_send_end = torch.cuda.Event(enable_timing=True)
        o_send_start.record()
        cross_result_on_gpu1 = cross_result.to('cuda:1', non_blocking=True)
        cross_result_on_gpu1.share_memory_()
        o_send_end.record()
        nvtx.range_pop()
        o_send_time_events.append((o_send_start, o_send_end))
        
        # 결과 전송
        cross_response_queue.put({
            'chunk_idx': chunk_idx,
            'result': cross_result_on_gpu1
        })
        
        # === Barrier: GPU1과 동기화 ===
        print(f"GPU0: Waiting at barrier for chunk {chunk_idx}")
        chunk_barriers[chunk_idx].wait()
        
        # GPU 작업 완료 대기
        torch.cuda.synchronize('cuda:0')
        
        # 시간 측정
        local_time = local_time_events[chunk_idx][0].elapsed_time(local_time_events[chunk_idx][1])
        q_receive_time = q_receive_time_events[chunk_idx][0].elapsed_time(q_receive_time_events[chunk_idx][1])
        cross_time = cross_time_events[chunk_idx][0].elapsed_time(cross_time_events[chunk_idx][1])
        o_send_time = o_send_time_events[chunk_idx][0].elapsed_time(o_send_time_events[chunk_idx][1])

        print(f"GPU0: Chunk {chunk_idx} - Local: {local_time:.3f}ms, Q Receive: {q_receive_time:.3f}ms, Cross: {cross_time:.3f}ms, O Send: {o_send_time:.3f}ms")
        
        nvtx.range_pop()
    
    total_end.record()
    
    nvtx.range_pop()
    
    # 모든 GPU 작업 완료 대기
    torch.cuda.synchronize()
    total_time = total_start.elapsed_time(total_end)
    
    # Event 쌍을 실제 시간값으로 변환
    local_times_ms = [start.elapsed_time(end) for start, end in local_time_events]
    q_receive_times_ms = [start.elapsed_time(end) for start, end in q_receive_time_events]
    o_send_times_ms = [start.elapsed_time(end) for start, end in o_send_time_events]
    cross_times_ms = [start.elapsed_time(end) for start, end in cross_time_events]
    
    # 결과 출력 형태 확인용
    output = torch.cat(local_results, dim=-2)
    print(f"GPU0: Output shape: {output.shape}")
    
    result_queue.put({
        'gpu_id': 0,
        'total_time': total_time,
        'local_times': local_times_ms,
        'q_receive_times': q_receive_times_ms,
        'o_send_times': o_send_times_ms,
        'cross_times': cross_times_ms
    })
    
    barrier_end.wait()
    print(f"GPU0: Completed in {total_time:.3f} ms")


def worker_gpu1(q_gpu1, k_gpu1, v_gpu1, cross_request_queue, cross_response_queue,
                result_queue, barrier_start, barrier_warmup, chunk_barriers, barrier_end):
    """
    GPU1 Worker (Stream 기반):
    각 청크마다:
    1. Stream 1: Local attention
    2. Stream 2: GPU0으로 q 보내기
    3. Stream 3: GPU0로부터 결과 받기
    4. 동기화 후 결과 합산
    5. Barrier로 GPU0과 동기화
    """
    torch.cuda.set_device(1)
    
    # Context Manager 초기화
    position_bias = load_position_bias(
        "/root/mwnoh/ReKV/model/attention/position_bias.pkl", device="cuda:1"
    )
    context_manager = ContextManager(
        position_bias, exc_block_size=exc_block_size, fattn=fattn
    )
    context_manager.init(
        batch_size=batch_size, num_heads=num_heads, dim_head=dim_head,
        dtype=torch.float16, device='cuda:1'
    )
    
    # 데이터를 GPU1로 이동
    q_gpu1 = q_gpu1.to('cuda:1')
    k_gpu1 = k_gpu1.to('cuda:1')
    v_gpu1 = v_gpu1.to('cuda:1')
    
    seq_len = q_gpu1.size(-2)
    num_chunks = (seq_len + chunk_size - 1) // chunk_size
    
    # Stream 생성
    stream_local = torch.cuda.Stream()
    stream_send = torch.cuda.Stream()
    stream_receive = torch.cuda.Stream()
    
    # 결과 저장
    local_results = []
    local_time_events = []
    q_send_time_events = []
    o_receive_time_events = []
    
    # 시작 동기화
    barrier_start.wait()
    
    nvtx.range_push("GPU1_Worker")
    
    # === Warmup Phase ===
    print("GPU1: Warmup phase...")
    nvtx.range_push("GPU1_Warmup")
    
    q_warmup = q_gpu1[:, :, 0:chunk_size, :]
    k_warmup = k_gpu1[:, :, 0:chunk_size, :]
    v_warmup = v_gpu1[:, :, 0:chunk_size, :]
    
    # Warmup: Local attention
    with torch.cuda.stream(stream_local):
        _ = context_manager._append(q_warmup, k_warmup, v_warmup)
    
    # Warmup: Send
    with torch.cuda.stream(stream_send):
        q_warmup_on_gpu0 = q_warmup.to('cuda:0', non_blocking=True)
        q_warmup_on_gpu0.share_memory_()
    stream_send.synchronize()
    
    cross_request_queue.put({
        'type': 'WARMUP',
        'chunk_idx': -1,
        'q_chunk': q_warmup_on_gpu0
    })
    
    # Warmup: Receive
    with torch.cuda.stream(stream_receive):
        try:
            warmup_response = cross_response_queue.get(timeout=10)
            _ = warmup_response['result']
        except queue.Empty:
            print("GPU1: Warmup receive failed")
            barrier_end.wait()
            return
    
    torch.cuda.synchronize('cuda:1')
    nvtx.range_pop()
    print("GPU1: Warmup completed")
    
    # Warmup 완료 동기화
    barrier_warmup.wait()
    
    total_start = torch.cuda.Event(enable_timing=True)
    total_end = torch.cuda.Event(enable_timing=True)
    total_start.record()
    
    print(f"GPU1: Processing {num_chunks} chunks with stream-based parallelism")
    
    for chunk_idx in range(num_chunks):
        nvtx.range_push(f"GPU1_Chunk{chunk_idx}")
        
        st = chunk_idx * chunk_size
        ed = min(st + chunk_size, seq_len)
        
        q_chunk = q_gpu1[:, :, st:ed, :]
        
        # === 병렬 실행: Send + Local (완전 독립) ===
        
        # Stream 1: GPU0으로 q 보내기
        nvtx.range_push(f"GPU1_Send_Chunk{chunk_idx}")
        q_send_start = torch.cuda.Event(enable_timing=True)
        q_send_end = torch.cuda.Event(enable_timing=True)
        
        with torch.cuda.stream(stream_send):
            q_send_start.record(stream_send)
            
            # GPU1→GPU0 직접 전송
            q_chunk_on_gpu0 = q_chunk.to('cuda:0', non_blocking=True)
            q_chunk_on_gpu0.share_memory_()
            
            q_send_end.record(stream_send)
        
        nvtx.range_pop()

        # Stream 2: Local attention (Send와 완전 독립적으로 실행)
        nvtx.range_push(f"GPU1_Local_Chunk{chunk_idx}")
        local_start = torch.cuda.Event(enable_timing=True)
        local_end = torch.cuda.Event(enable_timing=True)
        
        with torch.cuda.stream(stream_local):
            local_start.record(stream_local)
            
            k_causal = k_gpu1[:, :, 0:ed, :]
            v_causal = v_gpu1[:, :, 0:ed, :]
            
            local_result = context_manager._append(q_chunk, k_causal, v_causal)
            local_results.append(local_result)
            
            local_end.record(stream_local)
        
        nvtx.range_pop()
        
        # Send stream만 동기화 (local stream에 영향 없음)
        stream_send.synchronize()
        
        # Queue에 요청 전송 (CPU 작업, local과 완전 독립)
        cross_request_queue.put({
            'type': 'CROSS_ATTN',
            'chunk_idx': chunk_idx,
            'q_chunk': q_chunk_on_gpu0
        })
        
        # Stream 3: GPU0로부터 결과 받기
        nvtx.range_push(f"GPU1_Receive_Chunk{chunk_idx}")
        o_receive_start = torch.cuda.Event(enable_timing=True)
        o_receive_end = torch.cuda.Event(enable_timing=True)
        
        with torch.cuda.stream(stream_receive):
            o_receive_start.record(stream_receive)
            
            # Queue에서 결과 받기 (CPU 작업, stream과 무관)
            try:
                response = cross_response_queue.get(timeout=10)
                cross_result = response['result']
                recv_chunk_idx = response['chunk_idx']
                assert recv_chunk_idx == chunk_idx, f"Chunk mismatch: expected {chunk_idx}, got {recv_chunk_idx}"
            except queue.Empty:
                print(f"GPU1: Timeout waiting for cross response chunk {chunk_idx}")
                break
            
            o_receive_end.record(stream_receive)
        
        nvtx.range_pop()
        
        # === 동기화: 모든 Stream 완료 대기 ===
        torch.cuda.synchronize('cuda:1')
        
        local_time_events.append((local_start, local_end))
        q_send_time_events.append((q_send_start, q_send_end))
        o_receive_time_events.append((o_receive_start, o_receive_end))
        
        # === 결과 합산 ===
        nvtx.range_push(f"GPU1_Merge_Chunk{chunk_idx}")
        combined = local_results[chunk_idx] + cross_result
        local_results[chunk_idx] = combined
        nvtx.range_pop()
        
        # === Barrier: GPU0과 동기화 ===
        print(f"GPU1: Waiting at barrier for chunk {chunk_idx}")
        chunk_barriers[chunk_idx].wait()
        
        # GPU 작업 완료 대기
        torch.cuda.synchronize('cuda:1')
        
        # 시간 측정
        local_time = local_time_events[chunk_idx][0].elapsed_time(local_time_events[chunk_idx][1])
        q_send_time = q_send_time_events[chunk_idx][0].elapsed_time(q_send_time_events[chunk_idx][1])
        o_receive_time = o_receive_time_events[chunk_idx][0].elapsed_time(o_receive_time_events[chunk_idx][1])
        
        print(f"GPU1: Chunk {chunk_idx} - Local: {local_time:.3f}ms, Q Send: {q_send_time:.3f}ms, O Receive: {o_receive_time:.3f}ms")
        
        nvtx.range_pop()
    
    total_end.record()
    
    nvtx.range_pop()
    
    # 모든 GPU 작업 완료 대기
    torch.cuda.synchronize()
    total_time = total_start.elapsed_time(total_end)
    
    # Event 쌍을 실제 시간값으로 변환
    local_times_ms = [start.elapsed_time(end) for start, end in local_time_events]
    q_send_times_ms = [start.elapsed_time(end) for start, end in q_send_time_events]
    o_receive_times_ms = [start.elapsed_time(end) for start, end in o_receive_time_events]
    
    # 결과 출력 형태 확인용
    output = torch.cat(local_results, dim=-2)
    print(f"GPU1: Output shape: {output.shape}")
    
    result_queue.put({
        'gpu_id': 1,
        'total_time': total_time,
        'local_times': local_times_ms,
        'q_send_times': q_send_times_ms,
        'o_receive_times': o_receive_times_ms,
    })
    
    barrier_end.wait()
    print(f"GPU1: Completed in {total_time:.3f} ms")


def main():
    print("="*80)
    print("Multi-GPU Attention with Stream-Based Parallelism")
    print("="*80)
    print("Per-chunk execution:")
    print("  GPU0: Stream 1 (Local) || Stream 2 (Receive) → Cross (sequential)")
    print("  GPU1: Stream 1 (Local) || Stream 2 (Send) || Stream 3 (Receive) → Merge")
    print("  → Chunk-level barrier synchronization")
    print("="*80)
    print()
    
    torch.manual_seed(42)
    
    # 데이터 생성
    print("Generating data...")
    half_len = seq_length // 2
    
    q_gpu0 = torch.randn(batch_size, num_heads, half_len, dim_head, dtype=torch.float16)
    k_gpu0 = torch.randn(batch_size, num_heads, half_len, dim_head, dtype=torch.float16)
    v_gpu0 = torch.randn(batch_size, num_heads, half_len, dim_head, dtype=torch.float16)
    
    q_gpu1 = torch.randn(batch_size, num_heads, half_len, dim_head, dtype=torch.float16)
    k_gpu1 = torch.randn(batch_size, num_heads, half_len, dim_head, dtype=torch.float16)
    v_gpu1 = torch.randn(batch_size, num_heads, half_len, dim_head, dtype=torch.float16)
    
    print(f"GPU0 data: q.shape={q_gpu0.shape}")
    print(f"GPU1 data: q.shape={q_gpu1.shape}")
    print()
    
    # 프로세스 간 통신
    cross_request_queue = mp.Queue()
    cross_response_queue = mp.Queue()
    result_queue = mp.Queue()
    
    barrier_start = mp.Barrier(2)
    barrier_warmup = mp.Barrier(2)
    barrier_end = mp.Barrier(2)
    
    # 청크 단위 동기화를 위한 barrier 생성
    num_chunks = (half_len + chunk_size - 1) // chunk_size
    chunk_barriers = [mp.Barrier(2) for _ in range(num_chunks)]
    print(f"Created {num_chunks} chunk barriers for per-chunk synchronization")
    
    # 프로세스 생성
    print("Starting processes...")
    p0 = mp.Process(
        target=worker_gpu0,
        args=(q_gpu0, k_gpu0, v_gpu0, cross_request_queue, cross_response_queue,
              result_queue, barrier_start, barrier_warmup, chunk_barriers, barrier_end)
    )
    p1 = mp.Process(
        target=worker_gpu1,
        args=(q_gpu1, k_gpu1, v_gpu1, cross_request_queue, cross_response_queue,
              result_queue, barrier_start, barrier_warmup, chunk_barriers, barrier_end)
    )
    
    p0.start()
    p1.start()
    
    p0.join()
    p1.join()
    
    # 결과 수집
    results = {}
    while not result_queue.empty():
        result = result_queue.get()
        results[result['gpu_id']] = result
    
    # 통계 출력
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    for gpu_id in sorted(results.keys()):
        result = results[gpu_id]
        print(f"\nGPU {gpu_id}:")
        print(f"  Total time: {result['total_time']:.3f} ms")
        print(f"  Local attention chunks: {len(result['local_times'])}")
        print(f"  Average local time: {sum(result['local_times'])/len(result['local_times']):.3f} ms")
        
        if gpu_id == 0:
            print(f"  Send (receive) chunks: {len(result['o_send_times'])}")
            print(f"  Average O Send time: {sum(result['o_send_times'])/len(result['o_send_times']):.3f} ms")
            print(f"  Cross attention chunks: {len(result['cross_times'])}")
            print(f"  Average cross time: {sum(result['cross_times'])/len(result['cross_times']):.3f} ms")
        else:
            print(f"  Send chunks: {len(result['q_send_times'])}")
            print(f"  Average Q Send time: {sum(result['q_send_times'])/len(result['q_send_times']):.3f} ms")
            print(f"  Receive chunks: {len(result['o_receive_times'])}")
            print(f"  Average O Receive time: {sum(result['o_receive_times'])/len(result['o_receive_times']):.3f} ms")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()


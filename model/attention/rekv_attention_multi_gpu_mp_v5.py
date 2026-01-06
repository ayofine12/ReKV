import torch
import torch.multiprocessing as mp

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


def worker_gpu0(q_gpu0, k_gpu0, v_gpu0, 
                # Shared buffers for communication
                cross_q_buffer, cross_result_buffer, 
                cross_q_ready, cross_result_ready,
                result_queue, barrier_start, barrier_warmup, chunk_barriers, barrier_end):
    """
    GPU0 Worker (Shared Buffer 기반):
    - cross_q_buffer: GPU1에서 보낸 q (GPU0에 할당)
    - cross_result_buffer: GPU0에서 계산한 결과 (GPU1에서 읽음)
    - Flags로 동기화
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
    
    # Shared buffers를 GPU0로
    cross_q_buffer = cross_q_buffer.to('cuda:0')
    cross_result_buffer = cross_result_buffer.to('cuda:0')
    
    seq_len = q_gpu0.size(-2)
    num_chunks = (seq_len + chunk_size - 1) // chunk_size
    
    # Stream 생성
    stream_local = torch.cuda.Stream()
    stream_cross = torch.cuda.Stream()
    
    # 결과 저장
    local_results = []
    local_time_events = []
    cross_time_events = []
    
    # 시작 동기화
    barrier_start.wait()
    
    nvtx.range_push("GPU0_Worker")
    
    # === Warmup Phase ===
    print("GPU0: Warmup phase...")
    nvtx.range_push("GPU0_Warmup")
    
    q_warmup = q_gpu0[:, :, 0:chunk_size, :]
    k_warmup = k_gpu0[:, :, 0:chunk_size, :]
    v_warmup = v_gpu0[:, :, 0:chunk_size, :]
    _ = context_manager._append(q_warmup, k_warmup, v_warmup)
    
    # Warmup cross
    while not cross_q_ready[0].item():
        pass  # Spin wait
    q_warmup_from_gpu1 = cross_q_buffer[0]
    _ = context_manager._append(q_warmup_from_gpu1, k_gpu0, v_gpu0)
    cross_result_buffer[0].copy_(_)
    cross_result_ready[0].fill_(1)
    
    torch.cuda.synchronize('cuda:0')
    nvtx.range_pop()
    print("GPU0: Warmup completed")
    
    barrier_warmup.wait()
    
    total_start = torch.cuda.Event(enable_timing=True)
    total_end = torch.cuda.Event(enable_timing=True)
    total_start.record()
    
    print(f"GPU0: Processing {num_chunks} chunks with shared buffer")
    
    for chunk_idx in range(num_chunks):
        nvtx.range_push(f"GPU0_Chunk{chunk_idx}")
        
        st = chunk_idx * chunk_size
        ed = min(st + chunk_size, seq_len)
        
        # === Stream 1: Local attention ===
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
        
        # === Stream 2: Cross attention (병렬) ===
        nvtx.range_push(f"GPU0_Cross_Chunk{chunk_idx}")
        cross_start = torch.cuda.Event(enable_timing=True)
        cross_end = torch.cuda.Event(enable_timing=True)
        
        with torch.cuda.stream(stream_cross):
            cross_start.record(stream_cross)
            
            # GPU1이 데이터를 쓸 때까지 대기 (spin wait)
            while not cross_q_ready[chunk_idx + 1].item():  # +1 (warmup 제외)
                pass
            
            # Shared buffer에서 읽기
            q_chunk_from_gpu1 = cross_q_buffer[chunk_idx + 1]
            
            # Cross attention
            cross_result = context_manager._append(q_chunk_from_gpu1, k_gpu0, v_gpu0)
            
            # 결과를 shared buffer에 쓰기
            cross_result_buffer[chunk_idx + 1].copy_(cross_result)
            
            # GPU1에 알림
            cross_result_ready[chunk_idx + 1].fill_(1)
            
            cross_end.record(stream_cross)
        
        nvtx.range_pop()
        
        # === 동기화 ===
        torch.cuda.synchronize('cuda:0')
        
        local_time_events.append((local_start, local_end))
        cross_time_events.append((cross_start, cross_end))
        
        # Barrier
        print(f"GPU0: Waiting at barrier for chunk {chunk_idx}")
        chunk_barriers[chunk_idx].wait()
        
        torch.cuda.synchronize('cuda:0')
        
        local_time = local_time_events[chunk_idx][0].elapsed_time(local_time_events[chunk_idx][1])
        cross_time = cross_time_events[chunk_idx][0].elapsed_time(cross_time_events[chunk_idx][1])
        
        print(f"GPU0: Chunk {chunk_idx} - Local: {local_time:.3f}ms, Cross: {cross_time:.3f}ms")
        
        nvtx.range_pop()
    
    total_end.record()
    nvtx.range_pop()
    
    torch.cuda.synchronize()
    total_time = total_start.elapsed_time(total_end)
    
    local_times_ms = [start.elapsed_time(end) for start, end in local_time_events]
    cross_times_ms = [start.elapsed_time(end) for start, end in cross_time_events]
    
    output = torch.cat(local_results, dim=-2)
    print(f"GPU0: Output shape: {output.shape}")
    
    result_queue.put({
        'gpu_id': 0,
        'total_time': total_time,
        'local_times': local_times_ms,
        'cross_times': cross_times_ms
    })
    
    barrier_end.wait()
    print(f"GPU0: Completed in {total_time:.3f} ms")


def worker_gpu1(q_gpu1, k_gpu1, v_gpu1,
                # Shared buffers
                cross_q_buffer, cross_result_buffer,
                cross_q_ready, cross_result_ready,
                result_queue, barrier_start, barrier_warmup, chunk_barriers, barrier_end):
    """
    GPU1 Worker (Shared Buffer 기반)
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
    
    # Shared buffers (GPU1에서는 읽기/쓰기용)
    cross_result_buffer = cross_result_buffer.to('cuda:1')
    
    seq_len = q_gpu1.size(-2)
    num_chunks = (seq_len + chunk_size - 1) // chunk_size
    
    # Stream 생성
    stream_local = torch.cuda.Stream()
    stream_send = torch.cuda.Stream()
    stream_receive = torch.cuda.Stream()
    
    # 결과 저장
    local_results = []
    local_time_events = []
    send_time_events = []
    receive_time_events = []
    
    barrier_start.wait()
    
    nvtx.range_push("GPU1_Worker")
    
    # === Warmup ===
    print("GPU1: Warmup phase...")
    nvtx.range_push("GPU1_Warmup")
    
    q_warmup = q_gpu1[:, :, 0:chunk_size, :]
    k_warmup = k_gpu1[:, :, 0:chunk_size, :]
    v_warmup = v_gpu1[:, :, 0:chunk_size, :]
    _ = context_manager._append(q_warmup, k_warmup, v_warmup)
    
    # Warmup send (직접 전송, 단일 복사)
    cross_q_buffer[0].copy_(q_warmup, non_blocking=True)
    torch.cuda.synchronize('cuda:1')
    cross_q_ready[0].fill_(1)
    
    # Warmup receive
    while not cross_result_ready[0].item():
        pass
    _ = cross_result_buffer[0].to('cuda:1')
    
    torch.cuda.synchronize('cuda:1')
    nvtx.range_pop()
    print("GPU1: Warmup completed")
    
    barrier_warmup.wait()
    
    total_start = torch.cuda.Event(enable_timing=True)
    total_end = torch.cuda.Event(enable_timing=True)
    total_start.record()
    
    print(f"GPU1: Processing {num_chunks} chunks with shared buffer")
    
    for chunk_idx in range(num_chunks):
        nvtx.range_push(f"GPU1_Chunk{chunk_idx}")
        
        st = chunk_idx * chunk_size
        ed = min(st + chunk_size, seq_len)
        
        q_chunk = q_gpu1[:, :, st:ed, :]
        
        # === Stream 1: Local attention ===
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
        
        # === Stream 2: Send to GPU0 (병렬) ===
        nvtx.range_push(f"GPU1_Send_Chunk{chunk_idx}")
        send_start = torch.cuda.Event(enable_timing=True)
        send_end = torch.cuda.Event(enable_timing=True)
        
        with torch.cuda.stream(stream_send):
            send_start.record(stream_send)
            
            # Shared buffer에 직접 쓰기 (GPU1 → GPU0, 단일 복사)
            # copy_()가 자동으로 GPU 간 전송 처리
            cross_q_buffer[chunk_idx + 1].copy_(q_chunk, non_blocking=True)
            
            send_end.record(stream_send)
        
        # Send 완료 후 flag 설정
        stream_send.synchronize()
        cross_q_ready[chunk_idx + 1].fill_(1)
        
        nvtx.range_pop()
        
        # === Stream 3: Receive from GPU0 (병렬) ===
        nvtx.range_push(f"GPU1_Receive_Chunk{chunk_idx}")
        receive_start = torch.cuda.Event(enable_timing=True)
        receive_end = torch.cuda.Event(enable_timing=True)
        
        with torch.cuda.stream(stream_receive):
            receive_start.record(stream_receive)
            
            # GPU0가 결과를 쓸 때까지 대기
            while not cross_result_ready[chunk_idx + 1].item():
                pass
            
            # Shared buffer에서 읽기
            cross_result = cross_result_buffer[chunk_idx + 1].to('cuda:1', non_blocking=True)
            
            receive_end.record(stream_receive)
        
        nvtx.range_pop()
        
        # === 동기화 및 합산 ===
        torch.cuda.synchronize('cuda:1')
        
        local_time_events.append((local_start, local_end))
        send_time_events.append((send_start, send_end))
        receive_time_events.append((receive_start, receive_end))
        
        # 결과 합산
        combined = local_results[chunk_idx] + cross_result
        local_results[chunk_idx] = combined
        
        # Barrier
        print(f"GPU1: Waiting at barrier for chunk {chunk_idx}")
        chunk_barriers[chunk_idx].wait()
        
        torch.cuda.synchronize('cuda:1')
        
        local_time = local_time_events[chunk_idx][0].elapsed_time(local_time_events[chunk_idx][1])
        send_time = send_time_events[chunk_idx][0].elapsed_time(send_time_events[chunk_idx][1])
        receive_time = receive_time_events[chunk_idx][0].elapsed_time(receive_time_events[chunk_idx][1])
        
        print(f"GPU1: Chunk {chunk_idx} - Local: {local_time:.3f}ms, Send: {send_time:.3f}ms, Receive: {receive_time:.3f}ms")
        
        nvtx.range_pop()
    
    total_end.record()
    nvtx.range_pop()
    
    torch.cuda.synchronize()
    total_time = total_start.elapsed_time(total_end)
    
    local_times_ms = [start.elapsed_time(end) for start, end in local_time_events]
    send_times_ms = [start.elapsed_time(end) for start, end in send_time_events]
    receive_times_ms = [start.elapsed_time(end) for start, end in receive_time_events]
    
    output = torch.cat(local_results, dim=-2)
    print(f"GPU1: Output shape: {output.shape}")
    
    result_queue.put({
        'gpu_id': 1,
        'total_time': total_time,
        'local_times': local_times_ms,
        'send_times': send_times_ms,
        'receive_times': receive_times_ms
    })
    
    barrier_end.wait()
    print(f"GPU1: Completed in {total_time:.3f} ms")


def main():
    print("="*80)
    print("Multi-GPU Attention with Shared Buffer (No Queue)")
    print("="*80)
    print("Using shared GPU memory buffers for zero-copy communication")
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
    
    num_chunks = (half_len + chunk_size - 1) // chunk_size
    
    # Shared buffers 생성 (GPU 0에 할당, share_memory로 공유)
    print(f"Creating shared buffers for {num_chunks + 1} chunks (including warmup)...")
    cross_q_buffer = torch.zeros(num_chunks + 1, batch_size, num_heads, chunk_size, dim_head, 
                                   dtype=torch.float16, device='cuda:0')
    cross_result_buffer = torch.zeros(num_chunks + 1, batch_size, num_heads, chunk_size, dim_head,
                                       dtype=torch.float16, device='cuda:0')
    
    cross_q_buffer.share_memory_()
    cross_result_buffer.share_memory_()
    
    # Flags (CPU에 할당, multiprocessing shared)
    cross_q_ready = torch.zeros(num_chunks + 1, dtype=torch.int32).share_memory_()
    cross_result_ready = torch.zeros(num_chunks + 1, dtype=torch.int32).share_memory_()
    
    print("Shared buffers created")
    print()
    
    # 프로세스 간 통신
    result_queue = mp.Queue()
    
    barrier_start = mp.Barrier(2)
    barrier_warmup = mp.Barrier(2)
    barrier_end = mp.Barrier(2)
    
    chunk_barriers = [mp.Barrier(2) for _ in range(num_chunks)]
    print(f"Created {num_chunks} chunk barriers")
    
    # 프로세스 생성
    print("Starting processes...")
    p0 = mp.Process(
        target=worker_gpu0,
        args=(q_gpu0, k_gpu0, v_gpu0,
              cross_q_buffer, cross_result_buffer,
              cross_q_ready, cross_result_ready,
              result_queue, barrier_start, barrier_warmup, chunk_barriers, barrier_end)
    )
    p1 = mp.Process(
        target=worker_gpu1,
        args=(q_gpu1, k_gpu1, v_gpu1,
              cross_q_buffer, cross_result_buffer,
              cross_q_ready, cross_result_ready,
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
            print(f"  Cross attention chunks: {len(result['cross_times'])}")
            print(f"  Average cross time: {sum(result['cross_times'])/len(result['cross_times']):.3f} ms")
        else:
            print(f"  Send chunks: {len(result['send_times'])}")
            print(f"  Average send time: {sum(result['send_times'])/len(result['send_times']):.3f} ms")
            print(f"  Receive chunks: {len(result['receive_times'])}")
            print(f"  Average receive time: {sum(result['receive_times'])/len(result['receive_times']):.3f} ms")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()


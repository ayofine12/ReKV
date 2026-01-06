import torch
import torch.multiprocessing as mp
import threading
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
                result_queue, barrier_start, chunk_barriers, barrier_end):
    """
    GPU0 Worker:
    각 청크마다 로컬 attention과 cross attention을 동시에 처리
    각 청크 완료 후 GPU1과 동기화
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
    
    # 데이터를 GPU0로 이동
    q_gpu0 = q_gpu0.to('cuda:0')
    k_gpu0 = k_gpu0.to('cuda:0')
    v_gpu0 = v_gpu0.to('cuda:0')
    
    seq_len = q_gpu0.size(-2)
    num_chunks = (seq_len + chunk_size - 1) // chunk_size
    
    # 로컬 결과 저장
    local_results = []
    local_times = []
    cross_time_events = []  # Event 쌍 저장
    
    # Cross attention을 별도 스레드에서 처리
    cross_results = {}
    cross_queue = queue.Queue()
    
    def cross_attention_thread():
        """GPU1로부터 요청받은 cross attention을 처리하는 스레드"""
        while True:
            try:
                request = cross_request_queue.get(timeout=0.1)
                if request['type'] == 'DONE':
                    break
                
                chunk_idx = request['chunk_idx']
                q_chunk = request['q_chunk']  # 이미 GPU0에 있음 (GPU1에서 직접 전송됨)
                
                nvtx.range_push(f"GPU0_Cross_Chunk{chunk_idx}")
                
                cross_start = torch.cuda.Event(enable_timing=True)
                cross_end = torch.cuda.Event(enable_timing=True)
                cross_start.record()
                
                # Cross attention: GPU1의 q × GPU0의 전체 k,v
                cross_result = context_manager._append(q_chunk, k_gpu0, v_gpu0)
                
                # GPU0→GPU1 직접 전송 (비동기)
                cross_result_on_gpu1 = cross_result.to('cuda:1', non_blocking=True)
                cross_result_on_gpu1.share_memory_()
                
                cross_end.record()
                # synchronize 제거 - Event 쌍만 저장 (나중에 계산)
                
                nvtx.range_pop()
                
                cross_response_queue.put({
                    'chunk_idx': chunk_idx,
                    'result': cross_result_on_gpu1
                })
                
                # Event 쌍을 큐에 전달 (시간은 나중에 계산)
                cross_queue.put({'chunk_idx': chunk_idx, 'events': (cross_start, cross_end)})
                print(f"GPU0: Cross attention chunk {chunk_idx} queued, transferred to GPU1")
                
            except:
                continue
    
    # 시작 동기화
    barrier_start.wait()
    
    nvtx.range_push("GPU0_Worker")
    
    # Cross attention 스레드 시작
    cross_thread = threading.Thread(target=cross_attention_thread, daemon=True)
    cross_thread.start()
    
    total_start = torch.cuda.Event(enable_timing=True)
    total_end = torch.cuda.Event(enable_timing=True)
    total_start.record()
    
    # 로컬 attention 수행 (청크 단위로 동기화)
    print(f"GPU0: Processing {num_chunks} local chunks with per-chunk synchronization")
    for chunk_idx in range(num_chunks):
        st = chunk_idx * chunk_size
        ed = min(st + chunk_size, seq_len)
        
        nvtx.range_push(f"GPU0_Local_Chunk{chunk_idx}")
        
        local_start = torch.cuda.Event(enable_timing=True)
        local_end = torch.cuda.Event(enable_timing=True)
        local_start.record()
        
        q_chunk = q_gpu0[:, :, st:ed, :]
        k_causal = k_gpu0[:, :, 0:ed, :]  # causal
        v_causal = v_gpu0[:, :, 0:ed, :]
        
        local_result = context_manager._append(q_chunk, k_causal, v_causal)
        local_results.append(local_result)
        
        local_end.record()
        local_times.append((local_start, local_end))  # Event 쌍 저장
        
        # 청크 단위 동기화 - GPU1과 동기화
        print(f"GPU0: Waiting at barrier for chunk {chunk_idx}")
        chunk_barriers[chunk_idx].wait()
        print(f"GPU0: Barrier passed for chunk {chunk_idx}")
        
        # GPU 작업 완료 대기 (Barrier는 CPU만 동기화, GPU는 따로 필요)
        torch.cuda.synchronize('cuda:0')
        
        # Barrier 이후 시간 측정 및 NVTX 범위 종료
        local_time = local_times[chunk_idx][0].elapsed_time(local_times[chunk_idx][1])
        nvtx.range_pop()  # GPU0_Local_Chunk 종료 (실제 작업 완료 후)
        
        print(f"GPU0: Local chunk {chunk_idx} done ({local_time:.3f} ms)")
    
    # Cross attention 완료 대기
    cross_request_queue.put({'type': 'DONE'})
    cross_thread.join()
    
    # Cross attention Event 쌍 수집
    while not cross_queue.empty():
        item = cross_queue.get()
        cross_time_events.append(item['events'])
    
    total_end.record()
    
    nvtx.range_pop()
    
    output = torch.cat(local_results, dim=-2)
    
    # 모든 GPU 작업 완료 대기 (시간 측정 전에 필요)
    torch.cuda.synchronize()
    total_time = total_start.elapsed_time(total_end)
    
    # Event 쌍을 실제 시간값으로 변환
    local_times_ms = [start.elapsed_time(end) for start, end in local_times]
    cross_times_ms = [start.elapsed_time(end) for start, end in cross_time_events]
    
    result_queue.put({
        'gpu_id': 0,
        'output': output.cpu(),
        'total_time': total_time,
        'local_times': local_times_ms,
        'cross_times': cross_times_ms
    })
    
    barrier_end.wait()
    print(f"GPU0: Completed in {total_time:.3f} ms")


def worker_gpu1(q_gpu1, k_gpu1, v_gpu1, cross_request_queue, cross_response_queue,
                result_queue, barrier_start, chunk_barriers, barrier_end):
    """
    GPU1 Worker:
    각 청크마다:
    1. 로컬 attention 시작
    2. 동시에 GPU0에 cross attention 요청
    3. 모든 청크 완료 후 결과 수집 및 합산
    각 청크 완료 후 GPU0과 동기화
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
    
    local_results = []
    local_times = []
    transfer_times = []
    
    # Transfer 후 즉시 요청 전송하는 스레드
    def async_transfer_thread():
        """GPU1→GPU0 transfer 완료 후 GPU0에 요청 전송"""
        import time
        while True:
            item = transfer_queue.get()
            if item is None:  # 종료 신호
                break
                
            chunk_idx, transfer_event, q_chunk_on_gpu0 = item
            
            # Transfer 완료 대기 (폴링 방식으로 GPU1 방해 최소화)
            nvtx.range_push(f"GPU1_Transfer_Wait_Chunk{chunk_idx}")
            while not transfer_event.query():  # ← query()는 완료 여부만 확인 (블로킹 안 함)
                time.sleep(0.0001)  # 100us 대기 (CPU 양보)
            nvtx.range_pop()
            
            # Transfer 완료 후 GPU0에 요청 전송
            nvtx.range_push(f"GPU1_Send_Cross_Request_Chunk{chunk_idx}")
            cross_request_queue.put({
                'type': 'CROSS_ATTN',
                'chunk_idx': chunk_idx,
                'q_chunk': q_chunk_on_gpu0
            })
            nvtx.range_pop()
            print(f"GPU1: Transfer for chunk {chunk_idx} done, request sent to GPU0")
    
    transfer_queue = queue.Queue()
    transfer_thread = threading.Thread(target=async_transfer_thread, daemon=True)
    
    # 시작 동기화
    barrier_start.wait()
    
    nvtx.range_push("GPU1_Worker")
    
    transfer_thread.start()
    
    total_start = torch.cuda.Event(enable_timing=True)
    total_end = torch.cuda.Event(enable_timing=True)
    total_start.record()
    
    print(f"GPU1: Processing {num_chunks} chunks")
    
    # 각 청크 처리 (청크 단위로 동기화)
    print(f"GPU1: Processing {num_chunks} chunks with per-chunk synchronization")
    for chunk_idx in range(num_chunks):
        nvtx.range_push(f"GPU1_Chunk{chunk_idx}")
        
        st = chunk_idx * chunk_size
        ed = min(st + chunk_size, seq_len)
        
        q_chunk = q_gpu1[:, :, st:ed, :]
        
        # 병렬 실행을 위해: transfer 시작 → 즉시 local attention 시작
        transfer_start = torch.cuda.Event(enable_timing=True)
        transfer_end = torch.cuda.Event(enable_timing=True)
        local_start = torch.cuda.Event(enable_timing=True)
        local_end = torch.cuda.Event(enable_timing=True)
        
        # 1. GPU1→GPU0 직접 전송 시작 (비동기)
        nvtx.range_push(f"GPU1_Transfer_Chunk{chunk_idx}")
        transfer_start.record()
        
        # GPU 간 직접 전송 (비동기)
        q_chunk_on_gpu0 = q_chunk.to('cuda:0', non_blocking=True)
        # Shared memory로 만들어서 프로세스 간 공유 가능하게
        q_chunk_on_gpu0.share_memory_()
        
        transfer_end.record()
        nvtx.range_pop()
        
        # Transfer 완료 후 GPU0에 요청 전송 (별도 스레드에서 처리)
        transfer_queue.put((chunk_idx, transfer_end, q_chunk_on_gpu0))
        
        # 2. 즉시 로컬 attention 시작 (transfer와 병렬 실행)
        nvtx.range_push(f"GPU1_Local_Chunk{chunk_idx}")
        local_start.record()
        
        k_causal = k_gpu1[:, :, 0:ed, :]  # causal
        v_causal = v_gpu1[:, :, 0:ed, :]
        
        local_result = context_manager._append(q_chunk, k_causal, v_causal)
        local_results.append(local_result)
        
        local_end.record()
        
        # 3. Event 쌍 저장 (synchronize 제거 - barrier에서 어차피 기다림)
        local_times.append((local_start, local_end))
        transfer_times.append((transfer_start, transfer_end))
        
        # 청크 단위 동기화 - GPU0과 동기화
        print(f"GPU1: Waiting at barrier for chunk {chunk_idx}")
        chunk_barriers[chunk_idx].wait()
        print(f"GPU1: Barrier passed for chunk {chunk_idx}")
        
        # GPU 작업 완료 대기 (Barrier는 CPU만 동기화, GPU는 따로 필요)
        torch.cuda.synchronize('cuda:1')
        
        # Barrier 이후 시간 측정 및 NVTX 범위 종료
        local_time = local_times[chunk_idx][0].elapsed_time(local_times[chunk_idx][1])
        transfer_time = transfer_times[chunk_idx][0].elapsed_time(transfer_times[chunk_idx][1])
        nvtx.range_pop()  # GPU1_Local_Chunk 종료 (실제 작업 완료 후)
        
        print(f"GPU1: Chunk {chunk_idx} - Local: {local_time:.3f} ms, Transfer(GPU1→GPU0): {transfer_time:.3f} ms)")
        
        nvtx.range_pop()  # GPU1_Chunk 종료
    
    # Transfer 스레드 종료 신호 및 대기
    transfer_queue.put(None)
    transfer_thread.join()
    
    # GPU0에 종료 신호
    cross_request_queue.put({'type': 'DONE'})
    
    # 3. GPU0로부터 cross attention 결과 수신 및 합산
    nvtx.range_push("GPU1_MergeResults")
    
    print("GPU1: Waiting for cross attention results...")
    cross_results = {}
    for _ in range(num_chunks):
        response = cross_response_queue.get()
        chunk_idx = response['chunk_idx']
        cross_results[chunk_idx] = response['result']  # 이미 GPU1에 있음 (GPU0에서 직접 전송됨)
        print(f"GPU1: Received cross attention result for chunk {chunk_idx}")
    
    # Element-wise 합산
    final_results = []
    for chunk_idx in range(num_chunks):
        combined = local_results[chunk_idx] + cross_results[chunk_idx]
        final_results.append(combined)
    
    output = torch.cat(final_results, dim=-2)
    
    nvtx.range_pop()
    
    total_end.record()
    
    nvtx.range_pop()
    
    # 모든 GPU 작업 완료 대기 (시간 측정 전에 필요)
    torch.cuda.synchronize()
    total_time = total_start.elapsed_time(total_end)
    
    # Event 쌍을 실제 시간값으로 변환
    local_times_ms = [start.elapsed_time(end) for start, end in local_times]
    transfer_times_ms = [start.elapsed_time(end) for start, end in transfer_times]
    
    result_queue.put({
        'gpu_id': 1,
        'output': output.cpu(),
        'total_time': total_time,
        'local_times': local_times_ms,
        'transfer_times': transfer_times_ms
    })
    
    barrier_end.wait()
    print(f"GPU1: Completed in {total_time:.3f} ms")


def main():
    print("="*80)
    print("Multi-GPU Attention with True Parallelism (Multiprocessing + Threading)")
    print("="*80)
    print("Per-chunk parallel execution:")
    print("  GPU0: q[n*256:(n+1)*256] × k[0:(n+1)*256], v[0:(n+1)*256]")
    print("  GPU1: q[4096+n*256:4096+(n+1)*256] × k[4096:4096+(n+1)*256], v[...]")
    print("  GPU0: q[4096+n*256:4096+(n+1)*256] × k[0:4096], v[0:4096] (cross)")
    print("  → All three happen simultaneously!")
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
              result_queue, barrier_start, chunk_barriers, barrier_end)
    )
    p1 = mp.Process(
        target=worker_gpu1,
        args=(q_gpu1, k_gpu1, v_gpu1, cross_request_queue, cross_response_queue,
              result_queue, barrier_start, chunk_barriers, barrier_end)
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
        
        if gpu_id == 0 and 'cross_times' in result:
            print(f"  Cross attention chunks: {len(result['cross_times'])}")
            if result['cross_times']:
                print(f"  Average cross time: {sum(result['cross_times'])/len(result['cross_times']):.3f} ms")
        
        if gpu_id == 1 and 'transfer_times' in result:
            print(f"  Average transfer time: {sum(result['transfer_times'])/len(result['transfer_times']):.3f} ms")
        
        print(f"  Output shape: {result['output'].shape}")
    
    # 최종 출력
    output_gpu0 = results[0]['output']
    output_gpu1 = results[1]['output']
    final_output = torch.cat([output_gpu0, output_gpu1], dim=-2)
    
    print(f"\nFinal combined output shape: {final_output.shape}")
    print("="*80)
    print("Test completed successfully!")
    print("="*80)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()


import torch
import torch.multiprocessing as mp
import time

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


def worker_gpu0(q_gpu0, k_gpu0, v_gpu0, request_queue, response_queue, result_queue, barrier_start, barrier_end):
    """
    GPU0 Worker: 
    1. 자신의 q[0:4096]로 로컬 attention 수행
    2. 동시에 GPU1로부터 cross attention 요청 처리
    """
    torch.cuda.set_device(0)
    
    # Position bias 및 Context Manager 초기화
    position_bias = load_position_bias(
        "/root/mwnoh/ReKV/model/attention/position_bias.pkl",
        device="cuda:0"
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
    
    # 시작 동기화
    barrier_start.wait()
    
    nvtx.range_push("GPU0_Worker")
    
    seq_len = q_gpu0.size(-2)
    num_chunks = (seq_len + chunk_size - 1) // chunk_size
    
    print(f"GPU0: Processing {num_chunks} local chunks + handling cross-attention requests")
    
    local_results = []
    chunk_times = []
    
    total_start = torch.cuda.Event(enable_timing=True)
    total_end = torch.cuda.Event(enable_timing=True)
    total_start.record()
    
    # GPU0 로컬 attention 수행
    for chunk_idx in range(num_chunks):
        st = chunk_idx * chunk_size
        ed = min(st + chunk_size, seq_len)
        
        nvtx.range_push(f"GPU0_Local_Chunk{chunk_idx}")
        
        chunk_start = torch.cuda.Event(enable_timing=True)
        chunk_end = torch.cuda.Event(enable_timing=True)
        chunk_start.record()
        
        q_chunk = q_gpu0[:, :, st:ed, :]
        k_local = k_gpu0[:, :, 0:ed, :]  # causal
        v_local = v_gpu0[:, :, 0:ed, :]
        
        # 로컬 attention
        local_result = context_manager._append(q_chunk, k_local, v_local)
        local_results.append(local_result)
        
        chunk_end.record()
        torch.cuda.synchronize()
        chunk_time = chunk_start.elapsed_time(chunk_end)
        chunk_times.append(chunk_time)
        
        nvtx.range_pop()
        
        print(f"GPU0: Local chunk {chunk_idx} completed ({chunk_time:.3f} ms)")
        
        # 동시에 GPU1로부터 cross attention 요청 처리
        while not request_queue.empty():
            request = request_queue.get()
            
            if request['type'] == 'DONE':
                break
            
            req_chunk_idx = request['chunk_idx']
            q_chunk_cross = request['q_chunk'].to('cuda:0')
            
            nvtx.range_push(f"GPU0_CrossAttn_Chunk{req_chunk_idx}")
            
            # Cross attention
            cross_result = context_manager._append(q_chunk_cross, k_gpu0, v_gpu0)
            
            nvtx.range_pop()
            
            response_queue.put({
                'chunk_idx': req_chunk_idx,
                'result': cross_result.cpu()
            })
            
            print(f"GPU0: Processed cross-attention for chunk {req_chunk_idx}")
    
    # 남은 cross attention 요청 처리
    nvtx.range_push("GPU0_ProcessRemainingRequests")
    while True:
        if not request_queue.empty():
            request = request_queue.get()
            
            if request['type'] == 'DONE':
                break
            
            req_chunk_idx = request['chunk_idx']
            q_chunk_cross = request['q_chunk'].to('cuda:0')
            
            nvtx.range_push(f"GPU0_CrossAttn_Chunk{req_chunk_idx}")
            cross_result = context_manager._append(q_chunk_cross, k_gpu0, v_gpu0)
            nvtx.range_pop()
            
            response_queue.put({
                'chunk_idx': req_chunk_idx,
                'result': cross_result.cpu()
            })
            
            print(f"GPU0: Processed cross-attention for chunk {req_chunk_idx}")
        else:
            # 짧은 대기
            time.sleep(0.001)
            # GPU1이 완료했는지 확인
            if request_queue.empty():
                break
    nvtx.range_pop()
    
    # 최종 결과
    output = torch.cat(local_results, dim=-2)
    
    total_end.record()
    torch.cuda.synchronize()
    total_time = total_start.elapsed_time(total_end)
    
    nvtx.range_pop()
    
    result_queue.put({
        'gpu_id': 0,
        'output': output.cpu(),
        'total_time': total_time,
        'chunk_times': chunk_times
    })
    
    barrier_end.wait()
    print(f"GPU0: Completed in {total_time:.3f} ms")


def worker_gpu1(q_gpu1, k_gpu1, v_gpu1, request_queue, response_queue, result_queue, barrier_start, barrier_end):
    """
    GPU1 Worker: 
    1. 자신의 청크로 로컬 attention
    2. 동시에 GPU0에 cross attention 요청
    3. 결과를 받아서 합산
    """
    torch.cuda.set_device(1)
    
    # Position bias 및 Context Manager 초기화
    position_bias = load_position_bias(
        "/root/mwnoh/ReKV/model/attention/position_bias.pkl",
        device="cuda:1"
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
    
    # 시작 동기화
    barrier_start.wait()
    
    nvtx.range_push("GPU1_Worker")
    
    seq_len = q_gpu1.size(-2)
    num_chunks = (seq_len + chunk_size - 1) // chunk_size
    
    print(f"GPU1: Processing {num_chunks} chunks")
    
    local_results = []
    cross_results = {}
    chunk_times = []
    
    total_start = torch.cuda.Event(enable_timing=True)
    total_end = torch.cuda.Event(enable_timing=True)
    total_start.record()
    
    # 모든 청크 처리
    for chunk_idx in range(num_chunks):
        st = chunk_idx * chunk_size
        ed = min(st + chunk_size, seq_len)
        
        nvtx.range_push(f"GPU1_Chunk{chunk_idx}")
        
        chunk_start = torch.cuda.Event(enable_timing=True)
        chunk_end = torch.cuda.Event(enable_timing=True)
        chunk_start.record()
        
        q_chunk = q_gpu1[:, :, st:ed, :]
        k_local = k_gpu1[:, :, 0:ed, :]  # causal
        v_local = v_gpu1[:, :, 0:ed, :]
        
        # 1. GPU1 로컬 attention
        nvtx.range_push(f"GPU1_Local_Chunk{chunk_idx}")
        local_result = context_manager._append(q_chunk, k_local, v_local)
        nvtx.range_pop()
        
        # 2. GPU0에 cross attention 요청 (비동기)
        nvtx.range_push(f"GPU1_RequestCross_Chunk{chunk_idx}")
        request_queue.put({
            'type': 'CROSS_ATTN',
            'chunk_idx': chunk_idx,
            'q_chunk': q_chunk.cpu()  # GPU1 → CPU → GPU0
        })
        nvtx.range_pop()
        
        local_results.append(local_result)
        
        chunk_end.record()
        torch.cuda.synchronize()
        chunk_time = chunk_start.elapsed_time(chunk_end)
        chunk_times.append(chunk_time)
        
        nvtx.range_pop()
        
        print(f"GPU1: Chunk {chunk_idx} local attention completed ({chunk_time:.3f} ms)")
    
    # GPU0에게 완료 신호
    request_queue.put({'type': 'DONE'})
    
    # 3. GPU0로부터 cross attention 결과 수신 및 합산
    nvtx.range_push("GPU1_CollectAndMerge")
    
    print("GPU1: Waiting for cross-attention results from GPU0...")
    
    for _ in range(num_chunks):
        response = response_queue.get()
        chunk_idx = response['chunk_idx']
        cross_result = response['result'].to('cuda:1')  # CPU → GPU1
        cross_results[chunk_idx] = cross_result
        print(f"GPU1: Received cross-attention result for chunk {chunk_idx}")
    
    # 결과 합산 (element-wise)
    final_results = []
    for chunk_idx in range(num_chunks):
        combined = local_results[chunk_idx] + cross_results[chunk_idx]
        final_results.append(combined)
    
    output = torch.cat(final_results, dim=-2)
    
    nvtx.range_pop()
    
    total_end.record()
    torch.cuda.synchronize()
    total_time = total_start.elapsed_time(total_end)
    
    nvtx.range_pop()
    
    # 결과 반환
    result_queue.put({
        'gpu_id': 1,
        'output': output.cpu(),
        'total_time': total_time,
        'chunk_times': chunk_times
    })
    
    barrier_end.wait()
    print(f"GPU1: Completed in {total_time:.3f} ms")


def main():
    print("="*80)
    print("Multi-GPU Attention with Multiprocessing (Cross-GPU Collaboration)")
    print("="*80)
    print("Strategy:")
    print("  - GPU0: q[0:4096], k[0:4096], v[0:4096]")
    print("    * Local attention: q[0:4096] × k[0:4096], v[0:4096]")
    print("    * Cross attention: q[4096:8192] × k[0:4096], v[0:4096] (on request)")
    print()
    print("  - GPU1: q[4096:8192], k[4096:8192], v[4096:8192]")
    print("    * Local attention: q[4096:8192] × k[4096:8192], v[4096:8192]")
    print("    * Request cross attention from GPU0")
    print("    * Merge: local + cross (element-wise)")
    print()
    print("  Parallel execution:")
    print("    GPU0 local attention || GPU1 local attention || GPU0 cross attention")
    print("="*80)
    print()
    
    torch.manual_seed(42)
    
    # 데이터 생성 (CPU)
    print("Generating data...")
    half_len = seq_length // 2
    
    # GPU0용 (앞 절반)
    q_gpu0 = torch.randn(batch_size, num_heads, half_len, dim_head, dtype=torch.float16)
    k_gpu0 = torch.randn(batch_size, num_heads, half_len, dim_head, dtype=torch.float16)
    v_gpu0 = torch.randn(batch_size, num_heads, half_len, dim_head, dtype=torch.float16)
    
    # GPU1용 (뒤 절반)
    q_gpu1 = torch.randn(batch_size, num_heads, half_len, dim_head, dtype=torch.float16)
    k_gpu1 = torch.randn(batch_size, num_heads, half_len, dim_head, dtype=torch.float16)
    v_gpu1 = torch.randn(batch_size, num_heads, half_len, dim_head, dtype=torch.float16)
    
    print(f"GPU0 data: q.shape={q_gpu0.shape}, k.shape={k_gpu0.shape}, v.shape={v_gpu0.shape}")
    print(f"GPU1 data: q.shape={q_gpu1.shape}, k.shape={k_gpu1.shape}, v.shape={v_gpu1.shape}")
    print()
    
    # 프로세스 간 통신
    request_queue = mp.Queue()   # GPU1 → GPU0 요청
    response_queue = mp.Queue()  # GPU0 → GPU1 응답
    result_queue = mp.Queue()    # GPU1 → Main 최종 결과
    
    barrier_start = mp.Barrier(2)
    barrier_end = mp.Barrier(2)
    
    # 프로세스 생성
    print("Starting processes...")
    p0 = mp.Process(
        target=worker_gpu0,
        args=(q_gpu0, k_gpu0, v_gpu0, request_queue, response_queue, result_queue, barrier_start, barrier_end)
    )
    p1 = mp.Process(
        target=worker_gpu1,
        args=(q_gpu1, k_gpu1, v_gpu1, request_queue, response_queue, result_queue, barrier_start, barrier_end)
    )
    
    p0.start()
    p1.start()
    
    # 완료 대기
    p0.join()
    p1.join()
    
    # 결과 수집
    results = {}
    while not result_queue.empty():
        result = result_queue.get()
        results[result['gpu_id']] = result
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    for gpu_id in sorted(results.keys()):
        result = results[gpu_id]
        print(f"\nGPU {gpu_id}:")
        print(f"  Total time: {result['total_time']:.3f} ms")
        print(f"  Number of chunks: {len(result['chunk_times'])}")
        print(f"  Average chunk time: {sum(result['chunk_times']) / len(result['chunk_times']):.3f} ms")
        print(f"  Output shape: {result['output'].shape}")
    
    # 최종 출력 합치기
    output_gpu0 = results[0]['output']
    output_gpu1 = results[1]['output']
    final_output = torch.cat([output_gpu0, output_gpu1], dim=-2)
    
    print(f"\nFinal combined output shape: {final_output.shape}")
    print("="*80)
    print("\nTest completed successfully!")
    print("="*80)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()


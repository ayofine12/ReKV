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

# 설정 파라미터
batch_size = 1
num_heads = 32
dim_head = 128
seq_length = 8192
chunk_size = 256
exc_block_size = 256
fattn = True


def worker_gpu(gpu_id, q, k, v, result_queue, barrier_start):
    """각 GPU에서 attention을 수행하는 worker"""
    torch.cuda.set_device(gpu_id)
    
    # Position bias 로드
    position_bias = load_position_bias(
        "/root/mwnoh/ReKV/model/attention/position_bias.pkl",
        device=f"cuda:{gpu_id}"
    )
    
    # Context Manager 초기화
    context_manager = ContextManager(
        position_bias,
        exc_block_size=exc_block_size,
        fattn=fattn,
    )
    context_manager.init(
        batch_size=batch_size,
        num_heads=num_heads,
        dim_head=dim_head,
        dtype=torch.float16,
        device=f'cuda:{gpu_id}'
    )
    
    # 데이터를 해당 GPU로 이동
    q_gpu = q.to(f'cuda:{gpu_id}')
    k_gpu = k.to(f'cuda:{gpu_id}')
    v_gpu = v.to(f'cuda:{gpu_id}')
    
    # 시작 동기화 (모든 GPU가 동시에 시작)
    barrier_start.wait()
    
    nvtx.range_push(f"GPU{gpu_id}_Processing")
    
    seq_len = q_gpu.size(-2)
    num_chunks = (seq_len + chunk_size - 1) // chunk_size
    
    results = []
    chunk_times = []
    
    # Timing
    total_start = torch.cuda.Event(enable_timing=True)
    total_end = torch.cuda.Event(enable_timing=True)
    total_start.record()
    
    for chunk_idx in range(num_chunks):
        st = chunk_idx * chunk_size
        ed = min(st + chunk_size, seq_len)
        
        nvtx.range_push(f"GPU{gpu_id}_Chunk{chunk_idx}_q[{st}:{ed}]")
        
        chunk_start = torch.cuda.Event(enable_timing=True)
        chunk_end = torch.cuda.Event(enable_timing=True)
        chunk_start.record()
        
        q_chunk = q_gpu[:, :, st:ed, :]
        k_causal = k_gpu[:, :, 0:ed, :]  # causal
        v_causal = v_gpu[:, :, 0:ed, :]
        
        chunk_result = context_manager._append(q_chunk, k_causal, v_causal)
        
        chunk_end.record()
        torch.cuda.synchronize()
        
        results.append(chunk_result)
        chunk_time = chunk_start.elapsed_time(chunk_end)
        chunk_times.append(chunk_time)
        
        nvtx.range_pop()
    
    total_end.record()
    torch.cuda.synchronize()
    total_time = total_start.elapsed_time(total_end)
    
    nvtx.range_pop()
    
    # 결과 합치기
    output = torch.cat(results, dim=-2)
    
    # 결과를 CPU로 이동하여 반환
    result_queue.put({
        'gpu_id': gpu_id,
        'output': output.cpu(),
        'total_time': total_time,
        'chunk_times': chunk_times
    })
    
    print(f"✓ GPU {gpu_id} completed: {total_time:.3f} ms")


def main():
    print("="*80)
    print("Multi-GPU Attention with Multiprocessing")
    print("="*80)
    
    # Random seed
    torch.manual_seed(42)
    
    # 데이터 생성 (CPU에서)
    print("\nGenerating Q, K, V tensors on CPU...")
    half_len = seq_length // 2
    
    # GPU0용 데이터 (앞 절반)
    q_gpu0 = torch.randn(batch_size, num_heads, half_len, dim_head, dtype=torch.float16)
    k_gpu0 = torch.randn(batch_size, num_heads, half_len, dim_head, dtype=torch.float16)
    v_gpu0 = torch.randn(batch_size, num_heads, half_len, dim_head, dtype=torch.float16)
    
    # GPU1용 데이터 (뒤 절반)
    q_gpu1 = torch.randn(batch_size, num_heads, half_len, dim_head, dtype=torch.float16)
    k_gpu1 = torch.randn(batch_size, num_heads, half_len, dim_head, dtype=torch.float16)
    v_gpu1 = torch.randn(batch_size, num_heads, half_len, dim_head, dtype=torch.float16)
    
    print(f"GPU0 data: q.shape={q_gpu0.shape}")
    print(f"GPU1 data: q.shape={q_gpu1.shape}")
    
    # Result queue와 barrier
    result_queue = mp.Queue()
    barrier_start = mp.Barrier(2)
    
    # 프로세스 생성
    print("\nStarting parallel processing...")
    processes = []
    
    p0 = mp.Process(target=worker_gpu, args=(0, q_gpu0, k_gpu0, v_gpu0, result_queue, barrier_start))
    p1 = mp.Process(target=worker_gpu, args=(1, q_gpu1, k_gpu1, v_gpu1, result_queue, barrier_start))
    
    processes.append(p0)
    processes.append(p1)
    
    # 시작
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
    
    # 통계 출력
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
    
    print("\n" + "="*80)
    print("Test completed successfully!")
    print("="*80)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()


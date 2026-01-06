#!/usr/bin/env python
"""
Multiprocessing을 사용한 다중 GPU 병렬 실행 테스트
각 GPU가 독립적인 프로세스에서 실행됨
"""

import torch
import torch.multiprocessing as mp
import time

# NVTX
try:
    from torch.cuda import nvtx
    NVTX_AVAILABLE = True
except ImportError:
    NVTX_AVAILABLE = False
    print("Warning: NVTX not available")
    class DummyNVTX:
        @staticmethod
        def range_push(msg):
            pass
        @staticmethod
        def range_pop():
            pass
    nvtx = DummyNVTX()

def worker_gpu(gpu_id, num_ops, size, barrier_start, barrier_end):
    """각 GPU에서 실행될 worker 함수 - 똑같은 작업을 동시에 실행"""
    torch.cuda.set_device(gpu_id)
    
    # 데이터 생성 (동일한 seed로 동일한 데이터 생성)
    torch.manual_seed(42)  # 같은 seed로 재현 가능
    a = torch.randn(size, size, device=f'cuda:{gpu_id}', dtype=torch.float16)
    b = torch.randn(size, size, device=f'cuda:{gpu_id}', dtype=torch.float16)
    
    # Warmup
    for _ in range(2):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize()
    
    # 시작 동기화 (모든 프로세스가 정확히 동시에 시작)
    print(f"GPU {gpu_id} ready, waiting at barrier...")
    barrier_start.wait()
    print(f"GPU {gpu_id} starting work NOW!")
    
    # 타이밍 측정
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    nvtx.range_push(f"GPU{gpu_id}_IdenticalWork")
    start_event.record()
    
    # 똑같은 연산 수행
    nvtx.range_push(f"GPU{gpu_id}_MatMul_{num_ops}ops")
    results = []
    for i in range(num_ops):
        c = torch.matmul(a, b)
        results.append(c)
    nvtx.range_pop()
    
    # 동기화
    end_event.record()
    torch.cuda.synchronize()
    
    elapsed_time = start_event.elapsed_time(end_event)
    
    nvtx.range_pop()
    
    # 종료 동기화
    barrier_end.wait()
    
    print(f"✓ GPU {gpu_id} completed {num_ops} operations in {elapsed_time:.3f} ms")

def test_multiprocessing_parallel():
    """Multiprocessing을 사용한 병렬 실행 - 똑같은 작업을 동시에"""
    print("="*80)
    print("Test: Multiprocessing - Same Work on Different GPUs Simultaneously")
    print("="*80)
    print("Purpose: Verify GPU0 and GPU1 execute the SAME operations in parallel")
    print()
    
    num_gpus = 2
    num_ops = 30  # 더 많은 연산으로 확실하게 확인
    size = 4096
    
    # Barrier for synchronization
    barrier_start = mp.Barrier(num_gpus)
    barrier_end = mp.Barrier(num_gpus)
    
    print(f"Configuration:")
    print(f"  - GPUs: {num_gpus}")
    print(f"  - Matrix size: {size}x{size}")
    print(f"  - Number of operations per GPU: {num_ops}")
    print(f"  - Starting simultaneously (using barrier)")
    print()
    
    # 각 GPU마다 프로세스 생성
    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=worker_gpu,
            args=(gpu_id, num_ops, size, barrier_start, barrier_end)
        )
        processes.append(p)
        p.start()
    
    # 모든 프로세스 완료 대기
    for p in processes:
        p.join()
    
    print("\n✓ All processes completed")
    print("  Expected: GPU0 and GPU1 timelines should COMPLETELY OVERLAP")
    print("  This proves true parallel execution on different GPUs")
    print()

def worker_gpu_chunks(gpu_id, num_chunks, chunk_size, q_data, k_data, v_data, barrier_start, barrier_end):
    """청크 단위로 처리하는 worker (실제 attention 시나리오와 유사)"""
    torch.cuda.set_device(gpu_id)
    
    # 데이터를 GPU로 이동
    q = q_data.to(f'cuda:{gpu_id}')
    k = k_data.to(f'cuda:{gpu_id}')
    v = v_data.to(f'cuda:{gpu_id}')
    
    # 시작 동기화
    barrier_start.wait()
    
    nvtx.range_push(f"GPU{gpu_id}_ChunkProcessing")
    
    results = []
    for chunk_idx in range(num_chunks):
        st = chunk_idx * chunk_size
        ed = min(st + chunk_size, q.size(1))
        
        nvtx.range_push(f"GPU{gpu_id}_Chunk{chunk_idx}")
        
        # Attention-like operation (simplified)
        q_chunk = q[:, st:ed, :]
        k_chunk = k[:, :ed, :]  # causal
        v_chunk = v[:, :ed, :]
        
        # Scaled dot-product attention (simplified)
        attn_weights = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) / (k_chunk.size(-1) ** 0.5)
        attn_output = torch.matmul(attn_weights, v_chunk)
        
        results.append(attn_output)
        
        nvtx.range_pop()
    
    # 결과 합치기
    output = torch.cat(results, dim=1)
    
    torch.cuda.synchronize()
    
    nvtx.range_pop()
    
    # 종료 동기화
    barrier_end.wait()
    
    print(f"✓ GPU {gpu_id} completed {num_chunks} chunks")

def test_multiprocessing_chunks():
    """청크 단위 처리 테스트 (실제 attention과 유사)"""
    print("="*80)
    print("Test: Multiprocessing with Chunk-based Processing")
    print("="*80)
    
    num_gpus = 2
    seq_len = 4096
    chunk_size = 256
    num_heads = 32
    dim_head = 128
    
    num_chunks = seq_len // chunk_size
    
    # 데이터 생성 (CPU에서)
    q_data = torch.randn(1, seq_len, num_heads * dim_head, dtype=torch.float16)
    k_data = torch.randn(1, seq_len, num_heads * dim_head, dtype=torch.float16)
    v_data = torch.randn(1, seq_len, num_heads * dim_head, dtype=torch.float16)
    
    # Barrier
    barrier_start = mp.Barrier(num_gpus)
    barrier_end = mp.Barrier(num_gpus)
    
    # 각 GPU마다 프로세스 생성
    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=worker_gpu_chunks,
            args=(gpu_id, num_chunks, chunk_size, q_data, k_data, v_data, barrier_start, barrier_end)
        )
        processes.append(p)
        p.start()
    
    # 모든 프로세스 완료 대기
    for p in processes:
        p.join()
    
    print("\n✓ All chunk processing completed")
    print("  Expected: GPU0 and GPU1 chunk processing should overlap")
    print()

if __name__ == "__main__":
    # Multiprocessing 설정
    mp.set_start_method('spawn', force=True)
    
    print("\n" + "="*80)
    print("Multi-GPU Parallelism Test with Multiprocessing")
    print("="*80)
    print(f"NVTX Available: {NVTX_AVAILABLE}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print("="*80)
    print()
    
    # Tests
    test_multiprocessing_parallel()
    test_multiprocessing_chunks()
    
    print("="*80)
    print("All tests completed!")
    print("="*80)
    print("\nTo visualize with Nsight Systems:")
    print("  cd /root/mwnoh/ReKV")
    print("  eval \"$(conda shell.bash hook)\" && conda activate videoagent")
    print("  nsys profile --trace=cuda,nvtx \\")
    print("    --output=model/attention/nsys_reports/test_multiproc \\")
    print("    python model/attention/check_parallel_multiprocessing.py")
    print("\nWhat to look for in Nsight Systems:")
    print("  1. Find the 'GPU0_IdenticalWork' and 'GPU1_IdenticalWork' markers")
    print("  2. These two should START at the same time (barrier synchronization)")
    print("  3. These two should COMPLETELY OVERLAP in the timeline")
    print("  4. Both should END at approximately the same time")
    print("\nIf they overlap completely:")
    print("  ✓ TRUE parallel execution confirmed")
    print("  ✓ Multiprocessing enables independent GPU execution")
    print("\nIf they don't overlap:")
    print("  ✗ Check hardware/driver configuration")
    print("  ✗ May need to enable CUDA MPS or adjust settings")
    print("="*80)


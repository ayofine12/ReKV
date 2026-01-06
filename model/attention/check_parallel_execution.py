#!/usr/bin/env python
"""
간단한 테스트: 두 GPU에서 연산이 실제로 병렬 실행되는지 확인
"""

import torch
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

def test_parallel_streams_same_gpu():
    """같은 GPU 내에서 두 스트림의 병렬 실행 테스트"""
    print("="*80)
    print("Test 1: Same GPU, Different Streams")
    print("="*80)
    
    nvtx.range_push("Test1_Same_GPU")
    
    # 두 개의 독립적인 스트림 (같은 GPU)
    stream0 = torch.cuda.Stream(device='cuda:0')
    stream1 = torch.cuda.Stream(device='cuda:0')
    
    # 큰 행렬 곱셈
    size = 4096
    
    # Stream 0에서 실행
    with torch.cuda.stream(stream0):
        nvtx.range_push("Stream0_MatMul")
        a0 = torch.randn(size, size, device='cuda:0', dtype=torch.float16)
        b0 = torch.randn(size, size, device='cuda:0', dtype=torch.float16)
        for i in range(20):
            c0 = torch.matmul(a0, b0)
        nvtx.range_pop()
    
    # Stream 1에서 실행
    with torch.cuda.stream(stream1):
        nvtx.range_push("Stream1_MatMul")
        a1 = torch.randn(size, size, device='cuda:0', dtype=torch.float16)
        b1 = torch.randn(size, size, device='cuda:0', dtype=torch.float16)
        for i in range(20):
            c1 = torch.matmul(a1, b1)
        nvtx.range_pop()
    
    # 동기화
    stream0.synchronize()
    stream1.synchronize()
    
    nvtx.range_pop()
    
    print("✓ Test 1 completed")
    print("  Expected: May or may not overlap (depends on GPU utilization)")
    print()

def test_parallel_streams_different_gpu():
    """다른 GPU에서 병렬 실행 테스트"""
    print("="*80)
    print("Test 2: Different GPUs (Parallel Launch)")
    print("="*80)
    
    nvtx.range_push("Test2_Different_GPUs")
    
    # 각 GPU에 독립적인 스트림
    stream_gpu0 = torch.cuda.Stream(device='cuda:0')
    stream_gpu1 = torch.cuda.Stream(device='cuda:1')
    
    size = 4096
    
    # 데이터 미리 생성 (메인 스트림에서)
    a0 = torch.randn(size, size, device='cuda:0', dtype=torch.float16)
    b0 = torch.randn(size, size, device='cuda:0', dtype=torch.float16)
    a1 = torch.randn(size, size, device='cuda:1', dtype=torch.float16)
    b1 = torch.randn(size, size, device='cuda:1', dtype=torch.float16)
    
    # 모든 작업을 비동기로 큐잉 (synchronize 없이)
    with torch.cuda.stream(stream_gpu0):
        nvtx.range_push("GPU0_MatMul")
        for i in range(20):
            c0 = torch.matmul(a0, b0)
        nvtx.range_pop()
    
    with torch.cuda.stream(stream_gpu1):
        nvtx.range_push("GPU1_MatMul")
        for i in range(20):
            c1 = torch.matmul(a1, b1)
        nvtx.range_pop()
    
    # 이제 동기화 (한 번에)
    stream_gpu0.synchronize()
    stream_gpu1.synchronize()
    
    nvtx.range_pop()
    
    print("✓ Test 2 completed")
    print("  Expected: SHOULD overlap completely (different physical GPUs)")
    print()

def test_parallel_streams_different_gpu_v2():
    """다른 GPU에서 병렬 실행 테스트 - 버전 2 (완전 분리)"""
    print("="*80)
    print("Test 2b: Different GPUs (Separate Launch)")
    print("="*80)
    
    nvtx.range_push("Test2b_Different_GPUs_Separate")
    
    # 각 GPU에 독립적인 스트림
    stream_gpu0 = torch.cuda.Stream(device='cuda:0')
    stream_gpu1 = torch.cuda.Stream(device='cuda:1')
    
    size = 4096
    num_ops = 20
    
    # 데이터 미리 생성
    a0 = torch.randn(size, size, device='cuda:0', dtype=torch.float16)
    b0 = torch.randn(size, size, device='cuda:0', dtype=torch.float16)
    a1 = torch.randn(size, size, device='cuda:1', dtype=torch.float16)
    b1 = torch.randn(size, size, device='cuda:1', dtype=torch.float16)
    
    results_0 = []
    results_1 = []
    
    # GPU0 작업 모두 큐잉
    for i in range(num_ops):
        with torch.cuda.stream(stream_gpu0):
            if i == 0:
                nvtx.range_push("GPU0_All_MatMul")
            results_0.append(torch.matmul(a0, b0))
            if i == num_ops - 1:
                nvtx.range_pop()
    
    # GPU1 작업 모두 큐잉
    for i in range(num_ops):
        with torch.cuda.stream(stream_gpu1):
            if i == 0:
                nvtx.range_push("GPU1_All_MatMul")
            results_1.append(torch.matmul(a1, b1))
            if i == num_ops - 1:
                nvtx.range_pop()
    
    # 동기화
    stream_gpu0.synchronize()
    stream_gpu1.synchronize()
    
    nvtx.range_pop()
    
    print("✓ Test 2b completed")
    print("  Expected: GPU0 and GPU1 operations MUST overlap")
    print()

def test_sequential_launch():
    """순차적 실행 (비교용)"""
    print("="*80)
    print("Test 3: Sequential Launch on Different GPUs")
    print("="*80)
    
    nvtx.range_push("Test3_Sequential")
    
    stream_gpu0 = torch.cuda.Stream(device='cuda:0')
    stream_gpu1 = torch.cuda.Stream(device='cuda:1')
    
    size = 4096
    
    for i in range(10):
        # GPU0 먼저
        with torch.cuda.stream(stream_gpu0):
            nvtx.range_push(f"GPU0_Iter{i}")
            a0 = torch.randn(size, size, device='cuda:0', dtype=torch.float16)
            b0 = torch.randn(size, size, device='cuda:0', dtype=torch.float16)
            c0 = torch.matmul(a0, b0)
            nvtx.range_pop()
        
        # GPU1 다음
        with torch.cuda.stream(stream_gpu1):
            nvtx.range_push(f"GPU1_Iter{i}")
            a1 = torch.randn(size, size, device='cuda:1', dtype=torch.float16)
            b1 = torch.randn(size, size, device='cuda:1', dtype=torch.float16)
            c1 = torch.matmul(a1, b1)
            nvtx.range_pop()
    
    stream_gpu0.synchronize()
    stream_gpu1.synchronize()
    
    nvtx.range_pop()
    
    print("✓ Test 3 completed")
    print("  Expected: GPU0 and GPU1 tasks are interleaved (not overlapping)")
    print()

if __name__ == "__main__":
    print("\n" + "="*80)
    print("CUDA Stream Parallelism Test with NVTX")
    print("="*80)
    print(f"NVTX Available: {NVTX_AVAILABLE}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print("="*80)
    print()
    
    # Warmup
    print("Warming up...")
    torch.randn(100, 100, device='cuda:0').matmul(torch.randn(100, 100, device='cuda:0'))
    torch.randn(100, 100, device='cuda:1').matmul(torch.randn(100, 100, device='cuda:1'))
    torch.cuda.synchronize()
    print()
    
    # Tests
    test_parallel_streams_same_gpu()
    test_parallel_streams_different_gpu()
    test_parallel_streams_different_gpu_v2()
    test_sequential_launch()
    
    print("="*80)
    print("All tests completed!")
    print("="*80)
    print("\nTo visualize with Nsight Systems:")
    print("  nsys profile --trace=cuda,nvtx --output=test_parallel \\")
    print("    python model/attention/check_parallel_execution.py")
    print("\nLook for:")
    print("  - Test1: Streams on same GPU may or may not overlap")
    print("  - Test2: GPU0 and GPU1 SHOULD overlap (data prep in main stream)")
    print("  - Test2b: GPU0 and GPU1 MUST overlap completely (perfect separation)")
    print("  - Test3: GPU0 and GPU1 should NOT overlap (sequential)")
    print("\nIf Test2b doesn't show overlap, check:")
    print("  - Hardware/driver issue")
    print("  - CUDA context settings")
    print("  - Nsight Systems zoom level (it might be overlapping but too small to see)")
    print("="*80)


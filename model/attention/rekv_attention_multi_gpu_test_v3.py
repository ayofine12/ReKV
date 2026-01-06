import torch
import torch.nn.functional as F

from .kv_cache_manager_v3 import ContextManager
from .position_bias_utils import load_position_bias

# NVTX for profiling with Nsight Systems
try:
    from torch.cuda import nvtx
    NVTX_AVAILABLE = True
except ImportError:
    NVTX_AVAILABLE = False
    print("Warning: NVTX not available. Install with: pip install nvidia-pyindex && pip install nvidia-nvtx")
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
seq_length = 8192  # 전체 시퀀스 길이
chunk_size = 256  # q를 잘라서 처리할 크기
exc_block_size = 256  # ContextManager에서 사용하는 블록 크기
fattn = True

class MultiGPUAttentionPipeline:
    """
    8192 길이의 q, k, v를 GPU0과 GPU1에 각각 4096씩 나눠서 처리
    GPU0: q[0:4096], k[0:4096], v[0:4096]
    GPU1: q[4096:8192], k[4096:8192], v[4096:8192]
    
    각 GPU에서 256 크기로 q를 청크 단위로 처리하며,
    q만 필요에 따라 전송하고 각 GPU가 자신의 KV로 부분 attention 수행 후 결과 합산
    """
    def __init__(self, position_bias_gpu0, position_bias_gpu1):
        # 각 GPU별 ContextManager 생성
        self.context_manager_gpu0 = ContextManager(
            position_bias_gpu0,
            exc_block_size=exc_block_size,
            fattn=fattn,
        )
        
        self.context_manager_gpu1 = ContextManager(
            position_bias_gpu1,
            exc_block_size=exc_block_size,
            fattn=fattn,
        )
        
        # CUDA Stream 생성
        self.stream_gpu0_compute = torch.cuda.Stream(device='cuda:0')  # GPU0 로컬 attention
        self.stream_gpu0_cross = torch.cuda.Stream(device='cuda:0')     # GPU0 cross attention
        self.stream_gpu1_compute = torch.cuda.Stream(device='cuda:1')  # GPU1 로컬 attention
        self.stream_transfer_0to1 = torch.cuda.Stream(device='cuda:1')
        self.stream_transfer_1to0 = torch.cuda.Stream(device='cuda:0')
        
        self.initialized = False
    
    def init(self, dtype=torch.float16):
        """ContextManager 초기화"""
        self.context_manager_gpu0.init(
            batch_size=batch_size,
            num_heads=num_heads,
            dim_head=dim_head,
            dtype=dtype,
            device='cuda:0'
        )
        
        self.context_manager_gpu1.init(
            batch_size=batch_size,
            num_heads=num_heads,
            dim_head=dim_head,
            dtype=dtype,
            device='cuda:1'
        )
        
        self.initialized = True
    
    def parallel_attention(self, q_gpu0, k_gpu0, v_gpu0, q_gpu1, k_gpu1, v_gpu1):
        """
        병렬 attention 수행 (causal mask 적용)
        
        Args:
            q_gpu0, k_gpu0, v_gpu0: GPU0의 데이터 (길이 4096, 위치 0:4096)
            q_gpu1, k_gpu1, v_gpu1: GPU1의 데이터 (길이 4096, 위치 4096:8192)
        
        Returns:
            output_gpu0: GPU0의 attention 결과 (길이 4096), GPU0에 위치
            output_gpu1: GPU1의 attention 결과 (길이 4096), GPU1에 위치
            
        처리 방식:
        - GPU0의 q[0:4096]: GPU0의 k[0:4096]와만 attention (causal)
        - GPU1의 q[4096:8192]: GPU0의 k[0:4096] + GPU1의 k[4096:8192]와 attention (causal)
        """
        if not self.initialized:
            raise RuntimeError("Call init() first")
        
        len_gpu0 = q_gpu0.size(-2)
        len_gpu1 = q_gpu1.size(-2)
        num_chunks_gpu0 = (len_gpu0 + chunk_size - 1) // chunk_size
        num_chunks_gpu1 = (len_gpu1 + chunk_size - 1) // chunk_size
        
        results_gpu0 = []
        results_gpu1 = []
        
        # 시간 측정을 위한 리스트
        gpu0_local_attn_times = []      # GPU0 로컬 attention 시간
        gpu1_local_attn_times = []      # GPU1 로컬 attention 시간
        gpu0_cross_attn_times = []      # GPU0 cross attention 시간
        gpu1_to_gpu0_transfer_times = []  # GPU1 -> GPU0 q 전송 시간
        gpu0_to_gpu1_transfer_times = []  # GPU0 -> GPU1 결과 전송 시간
        
        print(f"\nStarting pipelined parallel attention:")
        print(f"GPU0: {num_chunks_gpu0} chunks, GPU1: {num_chunks_gpu1} chunks")
        print(f"Strategy: Overlap transfer with computation to hide latency")
        
        # ===== Pipelined parallel processing =====
        # 모든 청크를 비동기로 시작하여 병렬성 극대화
        max_chunks = max(num_chunks_gpu0, num_chunks_gpu1)
        
        # 각 청크의 이벤트와 중간 결과 저장
        chunk_data_gpu0 = []
        chunk_data_gpu1 = []
        
        # 전체 시간 측정 시작
        total_start = torch.cuda.Event(enable_timing=True)
        total_end = torch.cuda.Event(enable_timing=True)
        total_start.record()
        
        # ===== Phase 1: 모든 청크를 비동기로 시작 (synchronize 없이) =====
        print("\nPhase 1: Launching all computations asynchronously...")
        
        nvtx.range_push("Phase1_Launch_All_Chunks")
        
        for chunk_idx in range(max_chunks):
            chunk_events = {}
            
            # GPU0 로컬 attention (비동기 시작)
            if chunk_idx < num_chunks_gpu0:
                st_gpu0 = chunk_idx * chunk_size
                ed_gpu0 = min(st_gpu0 + chunk_size, len_gpu0)
                
                gpu0_local_start = torch.cuda.Event(enable_timing=True)
                gpu0_local_end = torch.cuda.Event(enable_timing=True)
                
                with torch.cuda.stream(self.stream_gpu0_compute):
                    nvtx.range_push(f"GPU0_Local_Chunk{chunk_idx}_q[{st_gpu0}:{ed_gpu0}]")
                    gpu0_local_start.record(self.stream_gpu0_compute)
                    
                    q_chunk_gpu0 = q_gpu0[:, :, st_gpu0:ed_gpu0, :]
                    k_local_gpu0 = k_gpu0[:, :, 0:ed_gpu0, :]  # causal
                    v_local_gpu0 = v_gpu0[:, :, 0:ed_gpu0, :]
                    
                    attn_result_gpu0 = self.context_manager_gpu0._append(
                        q_chunk_gpu0, k_local_gpu0, v_local_gpu0
                    )
                    
                    gpu0_local_end.record(self.stream_gpu0_compute)
                    nvtx.range_pop()
                
                chunk_events['gpu0_result'] = attn_result_gpu0
                chunk_events['gpu0_start'] = gpu0_local_start
                chunk_events['gpu0_end'] = gpu0_local_end
                chunk_data_gpu0.append(chunk_events.copy())
            
            # GPU1 처리: 로컬 attention + transfer + cross attention (모두 비동기)
            if chunk_idx < num_chunks_gpu1:
                st_gpu1 = chunk_idx * chunk_size
                ed_gpu1 = min(st_gpu1 + chunk_size, len_gpu1)
                
                q_chunk_gpu1 = q_gpu1[:, :, st_gpu1:ed_gpu1, :]
                
                gpu1_local_start = torch.cuda.Event(enable_timing=True)
                gpu1_local_end = torch.cuda.Event(enable_timing=True)
                gpu0_cross_start = torch.cuda.Event(enable_timing=True)
                gpu0_cross_end = torch.cuda.Event(enable_timing=True)
                transfer_1to0_start = torch.cuda.Event(enable_timing=True)
                transfer_1to0_end = torch.cuda.Event(enable_timing=True)
                
                # GPU1 로컬 attention (비동기)
                with torch.cuda.stream(self.stream_gpu1_compute):
                    nvtx.range_push(f"GPU1_Local_Chunk{chunk_idx}_q[{4096+st_gpu1}:{4096+ed_gpu1}]")
                    gpu1_local_start.record(self.stream_gpu1_compute)
                    
                    k_local_gpu1 = k_gpu1[:, :, 0:ed_gpu1, :]  # causal
                    v_local_gpu1 = v_gpu1[:, :, 0:ed_gpu1, :]
                    
                    local_attn_gpu1 = self.context_manager_gpu1._append(
                        q_chunk_gpu1, k_local_gpu1, v_local_gpu1
                    )
                    
                    gpu1_local_end.record(self.stream_gpu1_compute)
                    nvtx.range_pop()
                
                # GPU1 -> GPU0 transfer (비동기, 로컬 computation과 병렬)
                with torch.cuda.stream(self.stream_transfer_1to0):
                    nvtx.range_push(f"Transfer_GPU1->GPU0_Chunk{chunk_idx}")
                    transfer_1to0_start.record(self.stream_transfer_1to0)
                    
                    q_chunk_on_gpu0 = q_chunk_gpu1.to('cuda:0', non_blocking=True)
                    
                    transfer_1to0_end.record(self.stream_transfer_1to0)
                    nvtx.range_pop()
                
                # GPU0 cross attention (transfer 완료 후, GPU0 로컬과 병렬)
                with torch.cuda.stream(self.stream_gpu0_cross):
                    # transfer 완료 대기만 (GPU0 로컬과는 독립적)
                    self.stream_transfer_1to0.synchronize()
                    
                    nvtx.range_push(f"GPU0_Cross_Chunk{chunk_idx}_q[{4096+st_gpu1}:{4096+ed_gpu1}]")
                    gpu0_cross_start.record(self.stream_gpu0_cross)
                    
                    cross_attn_on_gpu0 = self.context_manager_gpu0._append(
                        q_chunk_on_gpu0,
                        k_gpu0,
                        v_gpu0
                    )
                    
                    gpu0_cross_end.record(self.stream_gpu0_cross)
                    nvtx.range_pop()
                
                chunk_events['local_attn_gpu1'] = local_attn_gpu1
                chunk_events['cross_attn_on_gpu0'] = cross_attn_on_gpu0
                chunk_events['gpu1_start'] = gpu1_local_start
                chunk_events['gpu1_end'] = gpu1_local_end
                chunk_events['cross_start'] = gpu0_cross_start
                chunk_events['cross_end'] = gpu0_cross_end
                chunk_events['transfer_1to0_start'] = transfer_1to0_start
                chunk_events['transfer_1to0_end'] = transfer_1to0_end
                chunk_data_gpu1.append(chunk_events.copy())
        
        nvtx.range_pop()  # Phase1_Launch_All_Chunks
        
        print(f"  Launched {len(chunk_data_gpu0)} chunks on GPU0")
        print(f"  Launched {len(chunk_data_gpu1)} chunks on GPU1")
        
        # ===== Phase 2: 결과 수집 (한 번에 동기화) =====
        print("\nPhase 2: Waiting for all computations to complete...")
        
        nvtx.range_push("Phase2_Synchronize")
        
        # 모든 스트림 동기화
        self.stream_gpu0_compute.synchronize()
        self.stream_gpu1_compute.synchronize()
        self.stream_gpu0_cross.synchronize()
        
        nvtx.range_pop()  # Phase2_Synchronize
        
        print("  All computations completed!")
        
        # GPU0 결과 수집
        print("\nPhase 3: Collecting results and measuring per-chunk timings...")
        nvtx.range_push("Phase3_Collect_Results")
        
        print("\n" + "="*80)
        print("GPU0 Chunks (Local Attention)")
        print("="*80)
        for chunk_idx, chunk_events in enumerate(chunk_data_gpu0):
            results_gpu0.append(chunk_events['gpu0_result'])
            
            gpu0_local_time = chunk_events['gpu0_start'].elapsed_time(chunk_events['gpu0_end'])
            gpu0_local_attn_times.append(gpu0_local_time)
            
            # KV 길이 계산 (causal이므로 청크마다 증가)
            q_start = chunk_idx * chunk_size
            q_end = min(q_start + chunk_size, len_gpu0)
            kv_length = q_end  # causal: 0부터 현재 위치까지
            
            print(f"Chunk {chunk_idx:2d}: q[{q_start:4d}:{q_end:4d}] × k,v[0:{kv_length:4d}] → {gpu0_local_time:.3f} ms")
        
        # GPU1 결과 수집 (cross attention 결과 전송 및 합산)
        print("\n" + "="*80)
        print("GPU1 Chunks (Local + Cross Attention)")
        print("="*80)
        for chunk_idx, chunk_events in enumerate(chunk_data_gpu1):
            transfer_0to1_start = torch.cuda.Event(enable_timing=True)
            transfer_0to1_end = torch.cuda.Event(enable_timing=True)
            
            # GPU0의 cross attention 결과를 GPU1로 전송
            with torch.cuda.stream(self.stream_transfer_0to1):
                transfer_0to1_start.record(self.stream_transfer_0to1)
                
                cross_attn_gpu1 = chunk_events['cross_attn_on_gpu0'].to('cuda:1')
                
                transfer_0to1_end.record(self.stream_transfer_0to1)
            
            self.stream_transfer_0to1.synchronize()
            
            # Element-wise 합산
            result_chunk_gpu1 = chunk_events['local_attn_gpu1'] + cross_attn_gpu1
            results_gpu1.append(result_chunk_gpu1)
            
            # 시간 기록
            gpu1_local_time = chunk_events['gpu1_start'].elapsed_time(chunk_events['gpu1_end'])
            gpu0_cross_time = chunk_events['cross_start'].elapsed_time(chunk_events['cross_end'])
            transfer_1to0_time = chunk_events['transfer_1to0_start'].elapsed_time(chunk_events['transfer_1to0_end'])
            transfer_0to1_time = transfer_0to1_start.elapsed_time(transfer_0to1_end)
            
            gpu1_local_attn_times.append(gpu1_local_time)
            gpu0_cross_attn_times.append(gpu0_cross_time)
            gpu1_to_gpu0_transfer_times.append(transfer_1to0_time)
            gpu0_to_gpu1_transfer_times.append(transfer_0to1_time)
            
            # GPU1 청크의 실제 위치와 KV 길이 계산
            q_start = 4096 + chunk_idx * chunk_size
            q_end = min(q_start + chunk_size, 4096 + len_gpu1)
            kv_length_gpu1 = chunk_idx * chunk_size + chunk_size  # GPU1 내에서의 KV 길이 (causal)
            kv_length_gpu0 = 4096  # GPU0의 전체 KV 길이
            
            print(f"Chunk {chunk_idx:2d}: q[{q_start:4d}:{q_end:4d}]")
            print(f"  ├─ GPU1 local: × k,v[4096:{4096+kv_length_gpu1:4d}] → {gpu1_local_time:.3f} ms")
            print(f"  ├─ GPU0 cross: × k,v[0:{kv_length_gpu0:4d}] → {gpu0_cross_time:.3f} ms")
            print(f"  ├─ Transfer 1→0: {transfer_1to0_time:.3f} ms")
            print(f"  └─ Transfer 0→1: {transfer_0to1_time:.3f} ms")
        
        # 전체 시간 측정 종료
        total_end.record()
        torch.cuda.synchronize()
        total_time = total_start.elapsed_time(total_end)
        
        nvtx.range_pop()  # Phase3_Collect_Results
        
        print(f"  Results collected!")
        
        # 최종 결과 합치기
        output_gpu0 = torch.cat(results_gpu0, dim=-2)  # (batch, num_heads, 4096, dim_head)
        output_gpu1 = torch.cat(results_gpu1, dim=-2)  # (batch, num_heads, 4096, dim_head)
        
        # 통계 출력
        print("\n" + "="*80)
        print("TIMING STATISTICS (Pipelined Parallel Execution)")
        print("="*80)
        
        print(f"\n🕒 Total execution time: {total_time:.3f} ms")
        
        if gpu0_local_attn_times:
            print(f"\nGPU0 Local Attention (q[0:4096] × k[0:4096]):")
            print(f"  - Number of chunks: {len(gpu0_local_attn_times)}")
            print(f"  - Average time: {sum(gpu0_local_attn_times) / len(gpu0_local_attn_times):.3f} ms")
            print(f"  - Min time: {min(gpu0_local_attn_times):.3f} ms")
            print(f"  - Max time: {max(gpu0_local_attn_times):.3f} ms")
            print(f"  - Total time: {sum(gpu0_local_attn_times):.3f} ms")
        
        if gpu1_local_attn_times:
            print(f"\nGPU1 Local Attention (q[4096:8192] × k[4096:8192]):")
            print(f"  - Number of chunks: {len(gpu1_local_attn_times)}")
            print(f"  - Average time: {sum(gpu1_local_attn_times) / len(gpu1_local_attn_times):.3f} ms")
            print(f"  - Min time: {min(gpu1_local_attn_times):.3f} ms")
            print(f"  - Max time: {max(gpu1_local_attn_times):.3f} ms")
            print(f"  - Total time: {sum(gpu1_local_attn_times):.3f} ms")
        
        if gpu0_cross_attn_times:
            print(f"\nGPU0 Cross Attention (q[4096:8192] × k[0:4096]):")
            print(f"  - Number of chunks: {len(gpu0_cross_attn_times)}")
            print(f"  - Average time: {sum(gpu0_cross_attn_times) / len(gpu0_cross_attn_times):.3f} ms")
            print(f"  - Min time: {min(gpu0_cross_attn_times):.3f} ms")
            print(f"  - Max time: {max(gpu0_cross_attn_times):.3f} ms")
            print(f"  - Total time: {sum(gpu0_cross_attn_times):.3f} ms")
        
        if gpu1_to_gpu0_transfer_times:
            print(f"\nTransfer GPU1 -> GPU0 (q transfer):")
            print(f"  - Number of transfers: {len(gpu1_to_gpu0_transfer_times)}")
            print(f"  - Average time: {sum(gpu1_to_gpu0_transfer_times) / len(gpu1_to_gpu0_transfer_times):.3f} ms")
            print(f"  - Min time: {min(gpu1_to_gpu0_transfer_times):.3f} ms")
            print(f"  - Max time: {max(gpu1_to_gpu0_transfer_times):.3f} ms")
            print(f"  - Total time: {sum(gpu1_to_gpu0_transfer_times):.3f} ms")
        
        if gpu0_to_gpu1_transfer_times:
            print(f"\nTransfer GPU0 -> GPU1 (result transfer):")
            print(f"  - Number of transfers: {len(gpu0_to_gpu1_transfer_times)}")
            print(f"  - Average time: {sum(gpu0_to_gpu1_transfer_times) / len(gpu0_to_gpu1_transfer_times):.3f} ms")
            print(f"  - Min time: {min(gpu0_to_gpu1_transfer_times):.3f} ms")
            print(f"  - Max time: {max(gpu0_to_gpu1_transfer_times):.3f} ms")
            print(f"  - Total time: {sum(gpu0_to_gpu1_transfer_times):.3f} ms")
        
        # 병렬 효율성 분석
        print("\n" + "-"*80)
        print("PARALLELISM ANALYSIS")
        print("-"*80)
        
        # 각 작업의 총 시간
        total_gpu0_local = sum(gpu0_local_attn_times) if gpu0_local_attn_times else 0
        total_gpu1_local = sum(gpu1_local_attn_times) if gpu1_local_attn_times else 0
        total_gpu0_cross = sum(gpu0_cross_attn_times) if gpu0_cross_attn_times else 0
        total_transfer_1to0 = sum(gpu1_to_gpu0_transfer_times) if gpu1_to_gpu0_transfer_times else 0
        total_transfer_0to1 = sum(gpu0_to_gpu1_transfer_times) if gpu0_to_gpu1_transfer_times else 0
        
        # 순차 실행 시 예상 시간
        sequential_time = total_gpu0_local + total_gpu1_local + total_gpu0_cross + total_transfer_1to0 + total_transfer_0to1
        
        # 병렬 효율성 (speedup)
        if sequential_time > 0:
            speedup = sequential_time / total_time
            efficiency = speedup / 2 * 100  # 2 GPUs
            
            print(f"\nSequential execution (estimated): {sequential_time:.3f} ms")
            print(f"  = GPU0 local ({total_gpu0_local:.3f})")
            print(f"  + GPU1 local ({total_gpu1_local:.3f})")  
            print(f"  + GPU0 cross ({total_gpu0_cross:.3f})")
            print(f"  + Transfer 1->0 ({total_transfer_1to0:.3f})")
            print(f"  + Transfer 0->1 ({total_transfer_0to1:.3f})")
            
            print(f"\nParallel execution (actual): {total_time:.3f} ms")
            print(f"🚀 Speedup: {speedup:.2f}x")
            print(f"📊 Parallel efficiency: {efficiency:.1f}%")
            
            # Transfer hiding 효과
            transfer_total = total_transfer_1to0 + total_transfer_0to1
            if transfer_total > 0:
                transfer_hidden_pct = (1 - (transfer_total / sequential_time)) * 100
                print(f"💾 Transfer latency hidden: ~{transfer_hidden_pct:.1f}% of total time")
        
        print("\n" + "="*80)
        print(f"Output GPU0 shape: {output_gpu0.shape}, device: {output_gpu0.device}")
        print(f"Output GPU1 shape: {output_gpu1.shape}, device: {output_gpu1.device}")
        print("="*80)
        
        return output_gpu0, output_gpu1


def make_qkv(batch_size, seq_len, num_heads, dim_head, device):
    """Q, K, V 텐서 생성"""
    q = torch.randn(batch_size, num_heads, seq_len, dim_head, dtype=torch.float16, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, dim_head, dtype=torch.float16, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, dim_head, dtype=torch.float16, device=device)
    return q, k, v


def main():
    print("="*80)
    print("Multi-GPU Parallel Attention Test v3")
    print("="*80)
    
    # Position bias 로드
    print("\nLoading position bias...")
    position_bias_gpu0 = load_position_bias("/root/mwnoh/ReKV/model/attention/position_bias.pkl", device="cuda:0")
    position_bias_gpu1 = load_position_bias("/root/mwnoh/ReKV/model/attention/position_bias.pkl", device="cuda:1")
    
    # Random seed 설정
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    # Pipeline 생성 및 초기화
    print("\nInitializing pipeline...")
    pipeline = MultiGPUAttentionPipeline(position_bias_gpu0, position_bias_gpu1)
    pipeline.init(dtype=torch.float16)
    
    # 데이터 생성 (8192를 4096씩 나눔)
    print("\nGenerating Q, K, V tensors...")
    half_len = seq_length // 2  # 4096
    
    # GPU0에 전반부 생성
    q_gpu0, k_gpu0, v_gpu0 = make_qkv(batch_size, half_len, num_heads, dim_head, 'cuda:0')
    print(f"GPU0 data: q.shape={q_gpu0.shape}")
    
    # GPU1에 후반부 생성
    q_gpu1, k_gpu1, v_gpu1 = make_qkv(batch_size, half_len, num_heads, dim_head, 'cuda:1')
    print(f"GPU1 data: q.shape={q_gpu1.shape}")
    
    # Warmup 단계
    print("\n" + "="*80)
    print("WARMUP PHASE (no timing)")
    print("="*80)
    
    # 1. Transfer warmup: GPU 간 데이터 전송 초기화
    print("1. Transfer warmup...")
    warmup_transfer_data = torch.randn(batch_size, num_heads, chunk_size, dim_head, dtype=torch.float16, device='cuda:0')
    
    # GPU0 -> GPU1 전송 warmup (여러 번)
    for _ in range(3):
        data_on_gpu1 = warmup_transfer_data.to('cuda:1', non_blocking=True)
        torch.cuda.synchronize()
    
    # GPU1 -> GPU0 전송 warmup (여러 번)
    warmup_transfer_data_gpu1 = torch.randn(batch_size, num_heads, chunk_size, dim_head, dtype=torch.float16, device='cuda:1')
    for _ in range(3):
        data_on_gpu0 = warmup_transfer_data_gpu1.to('cuda:0', non_blocking=True)
        torch.cuda.synchronize()
    
    print("   Transfer warmup completed!")
    
    # 2. Computation warmup: attention 연산 초기화
    print("2. Computation warmup...")
    warmup_len = chunk_size  # 실제 chunk 크기와 동일하게
    q_warmup_gpu0, k_warmup_gpu0, v_warmup_gpu0 = make_qkv(batch_size, warmup_len, num_heads, dim_head, 'cuda:0')
    q_warmup_gpu1, k_warmup_gpu1, v_warmup_gpu1 = make_qkv(batch_size, warmup_len, num_heads, dim_head, 'cuda:1')
    
    # warmup 실행 (출력 억제를 위해 일시적으로 변경)
    import sys
    from io import StringIO
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    # 여러 번 실행하여 충분히 warmup
    for _ in range(2):
        _ = pipeline.parallel_attention(
            q_warmup_gpu0, k_warmup_gpu0, v_warmup_gpu0,
            q_warmup_gpu1, k_warmup_gpu1, v_warmup_gpu1
        )
        torch.cuda.synchronize()
    
    sys.stdout = old_stdout
    print("   Computation warmup completed!")
    
    print("\nAll warmup completed!\n")
    
    # 타이밍 측정
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize()
    start_event.record()
    
    # 병렬 attention 수행
    print("="*80)
    print("ACTUAL EXECUTION (with timing)")
    print("="*80)
    output_gpu0, output_gpu1 = pipeline.parallel_attention(
        q_gpu0, k_gpu0, v_gpu0,
        q_gpu1, k_gpu1, v_gpu1
    )
    
    torch.cuda.synchronize()
    end_event.record()
    torch.cuda.synchronize()
    
    elapsed_time = start_event.elapsed_time(end_event)
    
    print("="*80)
    print(f"\nTotal execution time: {elapsed_time:.3f} ms")
    print(f"Output GPU0 shape: {output_gpu0.shape}, device: {output_gpu0.device}")
    print(f"Output GPU1 shape: {output_gpu1.shape}, device: {output_gpu1.device}")
    print("\nTest completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()


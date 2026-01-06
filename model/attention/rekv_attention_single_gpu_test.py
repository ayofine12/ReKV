import torch
import torch.nn.functional as F

from .kv_cache_manager_v3 import ContextManager
from .position_bias_utils import load_position_bias

# 설정 파라미터
batch_size = 1
num_heads = 32
dim_head = 128
seq_length = 8192  # 전체 시퀀스 길이
chunk_size = 256  # q를 잘라서 처리할 크기
exc_block_size = 256  # ContextManager에서 사용하는 블록 크기
fattn = True

class SingleGPUAttention:
    """
    단일 GPU에서 8192 길이의 q, k, v를 처리
    멀티 GPU 버전과 비교를 위한 baseline
    """
    def __init__(self, position_bias):
        self.context_manager = ContextManager(
            position_bias,
            exc_block_size=exc_block_size,
            fattn=fattn,
        )
        
        self.initialized = False
    
    def init(self, dtype=torch.float16, device='cuda:0'):
        """ContextManager 초기화"""
        self.context_manager.init(
            batch_size=batch_size,
            num_heads=num_heads,
            dim_head=dim_head,
            dtype=dtype,
            device=device
        )
        
        self.device = device
        self.initialized = True
    
    def attention(self, q, k, v):
        """
        단일 GPU에서 attention 수행
        
        Args:
            q, k, v: 입력 텐서 (길이 8192)
        
        Returns:
            attention 결과 (길이 8192)
        """
        if not self.initialized:
            raise RuntimeError("Call init() first")
        
        seq_len = q.size(-2)
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        
        results = []
        chunk_times = []
        
        # 전체 시간 측정 시작
        total_start = torch.cuda.Event(enable_timing=True)
        total_end = torch.cuda.Event(enable_timing=True)
        total_start.record()
        
        print(f"\nProcessing {num_chunks} chunks sequentially on single GPU...")
        
        # 각 청크 처리 (순차적)
        for chunk_idx in range(num_chunks):
            st = chunk_idx * chunk_size
            ed = min(st + chunk_size, seq_len)
            
            # 이벤트 생성 (시간 측정용)
            chunk_start = torch.cuda.Event(enable_timing=True)
            chunk_end = torch.cuda.Event(enable_timing=True)
            
            chunk_start.record()
            
            # Attention 계산 (causal)
            q_chunk = q[:, :, st:ed, :]
            k_causal = k[:, :, 0:ed, :]  # causal: 현재 위치까지만
            v_causal = v[:, :, 0:ed, :]
            
            chunk_result = self.context_manager._append(
                q_chunk, k_causal, v_causal
            )
            
            chunk_end.record()
            torch.cuda.synchronize()
            
            results.append(chunk_result)
            
            # 시간 기록
            chunk_time = chunk_start.elapsed_time(chunk_end)
            chunk_times.append(chunk_time)
            
            # KV 길이
            kv_length = ed
            print(f"Chunk {chunk_idx:2d}: q[{st:4d}:{ed:4d}] × k,v[0:{kv_length:4d}] → {chunk_time:.3f} ms")
        
        # 전체 시간 측정 종료
        total_end.record()
        torch.cuda.synchronize()
        total_time = total_start.elapsed_time(total_end)
        
        # 결과 합치기
        output = torch.cat(results, dim=-2)
        
        # 통계 출력
        self._print_statistics(total_time, chunk_times)
        
        return output
    
    def _print_statistics(self, total_time, chunk_times):
        """통계 출력"""
        print("\n" + "="*80)
        print("TIMING STATISTICS (Single GPU Sequential Execution)")
        print("="*80)
        
        print(f"\n🕒 Total execution time: {total_time:.3f} ms")
        
        if chunk_times:
            print(f"\nPer-chunk timing:")
            print(f"  - Number of chunks: {len(chunk_times)}")
            print(f"  - Average time: {sum(chunk_times) / len(chunk_times):.3f} ms")
            print(f"  - Min time: {min(chunk_times):.3f} ms")
            print(f"  - Max time: {max(chunk_times):.3f} ms")
            print(f"  - Total time (sum): {sum(chunk_times):.3f} ms")
            
            # 시간 증가 분석
            if len(chunk_times) > 1:
                time_increase = chunk_times[-1] - chunk_times[0]
                time_increase_pct = (time_increase / chunk_times[0]) * 100
                print(f"\n📈 Time increase (first → last chunk):")
                print(f"  - First chunk: {chunk_times[0]:.3f} ms")
                print(f"  - Last chunk: {chunk_times[-1]:.3f} ms")
                print(f"  - Increase: {time_increase:.3f} ms ({time_increase_pct:.1f}%)")
        
        print("\n" + "="*80)


def make_qkv(batch_size, seq_len, num_heads, dim_head, device):
    """Q, K, V 텐서 생성"""
    q = torch.randn(batch_size, num_heads, seq_len, dim_head, dtype=torch.float16, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, dim_head, dtype=torch.float16, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, dim_head, dtype=torch.float16, device=device)
    return q, k, v


def main():
    print("="*80)
    print("Single GPU Attention Test (Baseline)")
    print("="*80)
    
    # Position bias 로드
    print("\nLoading position bias...")
    position_bias = load_position_bias("/root/mwnoh/ReKV/model/attention/position_bias.pkl", device="cuda:0")
    
    # Random seed 설정
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    # SingleGPUAttention 생성 및 초기화
    print("\nInitializing single GPU attention...")
    attention_module = SingleGPUAttention(position_bias)
    attention_module.init(dtype=torch.float16, device='cuda:0')
    
    # 데이터 생성 (전체 8192)
    print(f"\nGenerating Q, K, V tensors (length: {seq_length})...")
    q, k, v = make_qkv(batch_size, seq_length, num_heads, dim_head, 'cuda:0')
    print(f"Data shape: q.shape={q.shape}")
    
    # Warmup 단계
    print("\n" + "="*80)
    print("WARMUP PHASE")
    print("="*80)
    
    print("1. Computation warmup...")
    warmup_len = chunk_size
    q_warmup, k_warmup, v_warmup = make_qkv(batch_size, warmup_len, num_heads, dim_head, 'cuda:0')
    
    # warmup 실행 (출력 억제)
    import sys
    from io import StringIO
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    for _ in range(2):
        _ = attention_module.attention(q_warmup, k_warmup, v_warmup)
        torch.cuda.synchronize()
    
    sys.stdout = old_stdout
    print("   Warmup completed!")
    
    # 실제 측정
    print("\n" + "="*80)
    print("ACTUAL EXECUTION")
    print("="*80)
    
    output = attention_module.attention(q, k, v)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Output device: {output.device}")
    print("\nTest completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()

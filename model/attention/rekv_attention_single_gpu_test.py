import torch
import threading
import os

from .kv_cache_manager import ContextManager
from .position_bias_utils import load_position_bias

def __main__():
    # Load position_bias for GPU 0
    position_bias_gpu0 = load_position_bias("/root/mwnoh/ReKV/model/attention/position_bias.pkl", device="cuda:0")
    
    n_init = 5
    n_local = 4096
    block_size = 256
    max_cached_block = 16
    topk = 64
    chunk_size = 32
    exc_block_size = 256
    fattn = True
    async_global_stream = True
    pin_memory = True

    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    # Create ContextManager for GPU 0
    past_key_value_gpu0 = ContextManager(
        position_bias_gpu0,
        n_init, n_local, 
        block_size, max_cached_block, topk, chunk_size, exc_block_size,
        fattn,
        async_global_stream,
        pin_memory,
    )

    batch_size = 1
    len_q = 4096
    num_heads = 32
    dim_head = 128

    h_q_gpu0 = torch.randn(batch_size, num_heads, n_init+len_q, dim_head, dtype=torch.float16, device="cuda:0")
    h_k_gpu0 = torch.randn(batch_size, num_heads, n_init+len_q, dim_head, dtype=torch.float16, device="cuda:0")
    h_v_gpu0 = torch.randn(batch_size, num_heads, n_init+len_q, dim_head, dtype=torch.float16, device="cuda:0")

    local_q_gpu0, local_k_gpu0, local_v_gpu0 = h_q_gpu0, h_k_gpu0, h_v_gpu0
    global_q_gpu0, global_k_gpu0, global_v_gpu0 = h_q_gpu0, h_k_gpu0, h_v_gpu0

    h_q_gpu1 = torch.randn(batch_size, num_heads, len_q, dim_head, dtype=torch.float16, device="cuda:0")
    h_k_gpu1 = torch.randn(batch_size, num_heads, len_q, dim_head, dtype=torch.float16, device="cuda:0")
    h_v_gpu1 = torch.randn(batch_size, num_heads, len_q, dim_head, dtype=torch.float16, device="cuda:0")

    local_q_gpu1, local_k_gpu1, local_v_gpu1 = h_q_gpu1, h_k_gpu1, h_v_gpu1
    global_q_gpu1, global_k_gpu1, global_v_gpu1 = h_q_gpu1, h_k_gpu1, h_v_gpu1

    h_q_gpu2 = torch.randn(batch_size, num_heads, len_q, dim_head, dtype=torch.float16, device="cuda:0")
    h_k_gpu2 = torch.randn(batch_size, num_heads, len_q, dim_head, dtype=torch.float16, device="cuda:0")
    h_v_gpu2 = torch.randn(batch_size, num_heads, len_q, dim_head, dtype=torch.float16, device="cuda:0")

    local_q_gpu2, local_k_gpu2, local_v_gpu2 = h_q_gpu2, h_k_gpu2, h_v_gpu2
    global_q_gpu2, global_k_gpu2, global_v_gpu2 = h_q_gpu2, h_k_gpu2, h_v_gpu2  

    o_gpu0 = past_key_value_gpu0.append(
        local_q_gpu0, local_k_gpu0, local_v_gpu0,
        global_q_gpu0, global_k_gpu0, global_v_gpu0,
    )

    h_q_gpu0 = torch.randn(batch_size, num_heads, len_q, dim_head, dtype=torch.float16, device="cuda:0")
    h_k_gpu0 = torch.randn(batch_size, num_heads, len_q, dim_head, dtype=torch.float16, device="cuda:0")
    h_v_gpu0 = torch.randn(batch_size, num_heads, len_q, dim_head, dtype=torch.float16, device="cuda:0")

    local_q_gpu0, local_k_gpu0, local_v_gpu0 = h_q_gpu0, h_k_gpu0, h_v_gpu0
    global_q_gpu0, global_k_gpu0, global_v_gpu0 = h_q_gpu0, h_k_gpu0, h_v_gpu0

    append_0_start = torch.cuda.Event(enable_timing=True)
    append_0_end = torch.cuda.Event(enable_timing=True)
    append_1_start = torch.cuda.Event(enable_timing=True)
    append_1_end = torch.cuda.Event(enable_timing=True)
    append_2_start = torch.cuda.Event(enable_timing=True)
    append_2_end = torch.cuda.Event(enable_timing=True)

    append_0_start.record()
    o_gpu0 = past_key_value_gpu0.append(
        local_q_gpu0, local_k_gpu0, local_v_gpu0,
        global_q_gpu0, global_k_gpu0, global_v_gpu0,
    )
    append_0_end.record()
    torch.cuda.synchronize()
    append_0_time_ms = append_0_start.elapsed_time(append_0_end)

    append_1_start = torch.cuda.Event(enable_timing=True)
    o_gpu1 = past_key_value_gpu0.append(
        local_q_gpu1, local_k_gpu1, local_v_gpu1,
        global_q_gpu1, global_k_gpu1, global_v_gpu1,
    )
    append_1_end.record()
    torch.cuda.synchronize()
    append_1_time_ms = append_1_start.elapsed_time(append_1_end)

    append_2_start = torch.cuda.Event(enable_timing=True)
    o_gpu2 = past_key_value_gpu0.append(
        local_q_gpu2, local_k_gpu2, local_v_gpu2,
        global_q_gpu2, global_k_gpu2, global_v_gpu2,
    )

    
    print(f"Append 0 time: {append_0_time_ms:.3f} ms")

if __name__ == "__main__":
    __main__()
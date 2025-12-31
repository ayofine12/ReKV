import torch
import threading
import os

from .kv_cache_manager import ContextManager
from .position_bias_utils import load_position_bias
from .projection_utils import load_projection

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

batch_size = 1
len_q = 4096
num_heads = 32
dim_head = 128
tensor_dim = 4

def transfer_src_to_dest(past_key_value_gpu_src, past_key_value_gpu_dest, dest_device):
    # Copy global remainder (updated during append)
    past_key_value_gpu_dest.global_remainder = (
        past_key_value_gpu_src.global_remainder[0].to(dest_device, non_blocking=True),
        past_key_value_gpu_src.global_remainder[1].to(dest_device, non_blocking=True),
    )

    past_key_value_gpu_dest._global_remainder_st = past_key_value_gpu_src._global_remainder_st
    past_key_value_gpu_dest._global_remainder_ed = past_key_value_gpu_src._global_remainder_ed
    
    # Copy init KV (may be updated during append)
    past_key_value_gpu_dest.init_k = past_key_value_gpu_src.init_k.to(dest_device, non_blocking=True)
    past_key_value_gpu_dest.init_v = past_key_value_gpu_src.init_v.to(dest_device, non_blocking=True)
    past_key_value_gpu_dest.init_exc = past_key_value_gpu_src.init_exc
    
    # Copy length (updated during append)
    past_key_value_gpu_dest.length = past_key_value_gpu_src.length
    
    # Copy num_global_block (increases when new blocks are created during append)
    past_key_value_gpu_dest.num_global_block = past_key_value_gpu_src.num_global_block
    
    # Copy global_blocks (CPU-based, can be shared)
    past_key_value_gpu_dest.global_blocks = past_key_value_gpu_src.global_blocks
    
    # Copy block_k data (representative keys) from GPU0 to GPU1
    for u in range(past_key_value_gpu_src.num_units):
        if past_key_value_gpu_src.block_k[u].length > 0:
            gpu0_data = past_key_value_gpu_src.block_k[u].get_data()  # (length, hidden_size)
            gpu1_data = gpu0_data.to("cuda:1", non_blocking=True)
            past_key_value_gpu_dest.block_k[u].append(gpu1_data)  # Append all at once

def make_query(batch_size, len_q, num_heads, dim_head, device):
    return torch.randn(batch_size, len_q, num_heads * dim_head, dtype=torch.float16, device=device)

def attention(len_q = 4096, past_key_value=None, current_query=None, prev_query=None, device="cuda:0"):
    projection_start = torch.cuda.Event(enable_timing=True)
    projection_end = torch.cuda.Event(enable_timing=True)
    append_start = torch.cuda.Event(enable_timing=True)
    append_end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    projection_start.record()

    project_q = load_projection("/root/mwnoh/ReKV/model/attention/projections/project_q.pkl", device=device)
    project_k = load_projection("/root/mwnoh/ReKV/model/attention/projections/project_k.pkl", device=device)
    project_v = load_projection("/root/mwnoh/ReKV/model/attention/projections/project_v.pkl", device=device)

    if current_query is not None:
        current_h_q = project_q(current_query)
        current_h_k = project_k(current_query)
        current_h_v = project_v(current_query)
        current_h_q = current_h_q.view(batch_size, len_q, num_heads, dim_head).permute(0, 2, 1, 3).contiguous()      # (batch, num_heads, len_q, dim_head)
        current_h_k = current_h_k.view(batch_size, len_q, num_heads, dim_head).permute(0, 2, 1, 3).contiguous()      # (batch, num_heads, len_q, dim_head)
        current_h_v = current_h_v.view(batch_size, len_q, num_heads, dim_head).permute(0, 2, 1, 3).contiguous()      # (batch, num_heads, len_q, dim_head)
        current_local_q, current_local_k, current_local_v = current_h_q, current_h_k, current_h_v
        current_global_q, current_global_k, current_global_v = current_h_q, current_h_k, current_h_v
    if prev_query is not None:
        prev_h_q = project_q(prev_query)
        prev_h_k = project_k(prev_query)
        prev_h_v = project_v(prev_query)
        prev_h_q = prev_h_q.view(batch_size, len_q, num_heads, dim_head).permute(0, 2, 1, 3).contiguous()          # (batch, num_heads, len_q, dim_head)
        prev_h_k = prev_h_k.view(batch_size, len_q, num_heads, dim_head).permute(0, 2, 1, 3).contiguous()          # (batch, num_heads, len_q, dim_head)
        prev_h_v = prev_h_v.view(batch_size, len_q, num_heads, dim_head).permute(0, 2, 1, 3).contiguous()          # (batch, num_heads, len_q, dim_head)
        prev_local_q, prev_local_k, prev_local_v = prev_h_q, prev_h_k, prev_h_v
        prev_global_q, prev_global_k, prev_global_v = prev_h_q, prev_h_k, prev_h_v    



    torch.cuda.synchronize()
    projection_end.record()
    projection_time_ms = projection_start.elapsed_time(projection_end)
    print(f"Projection time: {projection_time_ms:.3f} ms")

    if prev_query is not None:
        o = past_key_value.append(
            prev_local_q, prev_local_k, prev_local_v,
            prev_global_q, prev_global_k, prev_global_v,
        )

    torch.cuda.synchronize()
    append_start.record()
    if current_query is not None:
        o = past_key_value.append(
            current_local_q, current_local_k, current_local_v,
            current_global_q, current_global_k, current_global_v,
        )
    
    torch.cuda.synchronize()
    append_end.record()
    append_time_ms = append_start.elapsed_time(append_end)
    print(f"Append time: {append_time_ms:.3f} ms")

    return past_key_value

def __main__():
    
    position_bias_gpu0 = load_position_bias("/root/mwnoh/ReKV/model/attention/position_bias.pkl", device="cuda:0")
    position_bias_gpu1 = load_position_bias("/root/mwnoh/ReKV/model/attention/position_bias.pkl", device="cuda:1")
    position_bias_gpu2 = load_position_bias("/root/mwnoh/ReKV/model/attention/position_bias.pkl", device="cuda:2")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    past_key_value_gpu0 = ContextManager(
        position_bias_gpu0,
        n_init, n_local, 
        block_size, max_cached_block, topk, chunk_size, exc_block_size,
        fattn,
        async_global_stream,
        pin_memory,
    )
    past_key_value_gpu0.init(batch_size, num_heads, dim_head, tensor_dim, torch.float16, "cuda:0")

    past_key_value_gpu1 = ContextManager(
        position_bias_gpu1,
        n_init, n_local, 
        block_size, max_cached_block, topk, chunk_size, exc_block_size,
        fattn,
        async_global_stream,
        pin_memory,
    )
    past_key_value_gpu1.init(batch_size, num_heads, dim_head, tensor_dim, torch.float16, "cuda:1")

    past_key_value_gpu2 = ContextManager(
        position_bias_gpu2,
        n_init, n_local, 
        block_size, max_cached_block, topk, chunk_size, exc_block_size,
        fattn,
        async_global_stream,
        pin_memory,
    )
    past_key_value_gpu2.init(batch_size, num_heads, dim_head, tensor_dim, torch.float16, "cuda:2")

    query0_gpu0 = make_query(batch_size, n_init+len_q, num_heads, dim_head, "cuda:0")
    query0_gpu1 = make_query(batch_size, len_q, num_heads, dim_head, "cuda:1")
    query1_gpu1 = make_query(batch_size, len_q, num_heads, dim_head, "cuda:1")
    query1_gpu2 = make_query(batch_size, len_q, num_heads, dim_head, "cuda:2")
    query2_gpu2 = make_query(batch_size, len_q, num_heads, dim_head, "cuda:2")

    
    past_key_value_gpu0 = attention(n_init+len_q, past_key_value_gpu0, query0_gpu0, None, "cuda:0")
    transfer_src_to_dest(past_key_value_gpu0, past_key_value_gpu1, "cuda:1")

    past_key_value_gpu1 = attention(len_q, past_key_value_gpu1, query0_gpu1, query1_gpu1, "cuda:1")
    transfer_src_to_dest(past_key_value_gpu1, past_key_value_gpu2, "cuda:2")

    past_key_value_gpu2 = attention(len_q, past_key_value_gpu2, query1_gpu2, query2_gpu2, "cuda:2")
    
if __name__ == "__main__":
    __main__()
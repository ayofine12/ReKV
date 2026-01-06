import torch

from .kv_cache_manager_v3 import ContextManager
from .position_bias_utils import load_position_bias
from .projection_utils import load_projection

exc_block_size = 256
max_len = 4096
fattn = True

batch_size = 1
num_heads = 32
dim_head = 128

def make_query(batch_size, len_q, num_heads, dim_head, device):
    return torch.randn(batch_size, len_q, num_heads * dim_head, dtype=torch.float16, device=device)

def attention(q, k, v, past_key_value=None, device="cuda:0"):
    project_q = load_projection("/root/mwnoh/ReKV/model/attention/projections/project_q.pkl", device=device)
    project_k = load_projection("/root/mwnoh/ReKV/model/attention/projections/project_k.pkl", device=device)
    project_v = load_projection("/root/mwnoh/ReKV/model/attention/projections/project_v.pkl", device=device)

    h_q = project_q(q)
    h_k = project_k(k)
    h_v = project_v(v)

    h_q = h_q.view(batch_size, -1, num_heads, dim_head).permute(0, 2, 1, 3).contiguous()
    h_k = h_k.view(batch_size, -1, num_heads, dim_head).permute(0, 2, 1, 3).contiguous()
    h_v = h_v.view(batch_size, -1, num_heads, dim_head).permute(0, 2, 1, 3).contiguous()

    o = past_key_value.append(
        h_q, h_k, h_v,
    )

    return o

def __main__():
    position_bias_gpu = load_position_bias("/root/mwnoh/ReKV/model/attention/position_bias.pkl", device="cuda:0")

    past_key_value_gpu = ContextManager(
        position_bias_gpu,
        exc_block_size,
        fattn,
    )
    past_key_value_gpu.init(batch_size, num_heads, dim_head, torch.float16, "cuda:0")

    attention_start = torch.cuda.Event(enable_timing=True)
    attention_end = torch.cuda.Event(enable_timing=True)

    # q = 256, k = 256, v = 256
    q = make_query(batch_size, exc_block_size, num_heads, dim_head, "cuda:0")   
    k = v = make_query(batch_size, exc_block_size, num_heads, dim_head, "cuda:0")

    # warmup
    for i in range(3):
        o = attention(q, k, v, past_key_value_gpu, "cuda:0")

    torch.cuda.synchronize()
    attention_start.record()
    o = attention(q, k, v, past_key_value_gpu, "cuda:0")
    torch.cuda.synchronize()
    attention_end.record()
    attention_time = attention_start.elapsed_time(attention_end)
    print(f"Small Attention time: {attention_time} ms")

    # q = 256, k = 4096, v = 4096
    q = make_query(batch_size, exc_block_size, num_heads, dim_head, "cuda:0")   
    k = v = make_query(batch_size, max_len, num_heads, dim_head, "cuda:0")

    # warmup
    for i in range(3):
        o = attention(q, k, v, past_key_value_gpu, "cuda:0")

    torch.cuda.synchronize()
    attention_start.record()
    o = attention(q, k, v, past_key_value_gpu, "cuda:0")
    torch.cuda.synchronize()
    attention_end.record()
    attention_time = attention_start.elapsed_time(attention_end)
    print(f"LargeAttention time: {attention_time} ms")

    transfer_start = torch.cuda.Event(enable_timing=True)
    transfer_end = torch.cuda.Event(enable_timing=True)

    # warmup
    for i in range(3):
        q_gpu1= q.to("cuda:1")

    torch.cuda.synchronize()
    transfer_start.record() 
    q_gpu1 = q.to("cuda:1")
    torch.cuda.synchronize()
    transfer_end.record()
    transfer_time = transfer_start.elapsed_time(transfer_end)
    print(f"Transfer time: {transfer_time} ms")

if __name__ == "__main__":
    __main__()
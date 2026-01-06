import torch

from .dot_production_attention import get_multi_stage_dot_production_attention

class ContextManager:
    def __init__(self,
                 position_embedding,
                 exc_block_size,
                 fattn: bool = False,
                #  async_migration_stream: bool = False,
    ):
        self.position_embedding = position_embedding
        self.exc_block_size = exc_block_size
        self.fattn = fattn
        self.initialized = False
        # self.async_migration_stream = async_migration_stream
        self.Attn, _ = get_multi_stage_dot_production_attention(fattn)
        # if self.async_migration_stream:
        #     self.migration_stream = torch.cuda.Stream()
        # else:
        #     self.migration_stream = None

    def init(
        self,
        batch_size, 
        num_heads, 
        dim_head,
        dtype,
        device,
    ):  
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.dtype = dtype
        self.device = device
        self.local_k = torch.empty((self.batch_size, self.num_heads, 0, self.dim_head), dtype=self.dtype, device=self.device)
        self.local_v = torch.empty((self.batch_size, self.num_heads, 0, self.dim_head), dtype=self.dtype, device=self.device)
        self.length = 0
        self.initialized = True

    def _append(
        self,
        local_q, local_k, local_v,
    ):
        """calculate attention results 

        Args:
            local_q (_type_): (batch_size, num_heads, length, dim_head)
            local_k (_type_): (batch_size, num_heads, length, dim_head)
            local_v (_type_): (batch_size, num_heads, length, dim_head)

        Returns:
            chunk_o: (batch_size, num_heads, length, dim_head)
        """
        local_h_q, local_h_k = self.position_embedding(local_q, local_k)
        local_h_v = local_v

        attn = self.Attn(local_h_q.shape, local_h_q.dtype, local_h_q.device)
        attn.append(
            local_h_q, local_h_k, local_h_v, 
            end=True, get_score=False, sliding_window=None
        )

        o, _ = attn.get_result()

        return o.view((self.batch_size, self.num_heads, -1, self.dim_head))

    def append(self,
        local_q, local_k, local_v,
    ):
        input_length = local_q.size(-2)

        # append local KV
        # self.local_k = torch.cat((self.local_k, local_k), dim=-2)
        # self.local_v = torch.cat((self.local_v, local_v), dim=-2)
        self.local_k = local_k
        self.local_v = local_v
        kv_length = self.local_k.size(-2)

        o_list = []
        for st in range(0, input_length, self.exc_block_size):  # Process the input tokens in blocks.
            ed = min(st + self.exc_block_size, input_length)

            # calculate attention results
            # kv_st = max(kv_length + st - input_length, 0)
            kv_st = 0
            kv_ed = kv_length + ed - input_length
            chunk_o = self._append(
                local_q[:, :, st:ed, :],
                self.local_k[:, :, kv_st: kv_ed, :],
                self.local_v[:, :, kv_st: kv_ed, :],
            )
            o_list.append(chunk_o)
            
        self.length += input_length

        ret = torch.cat(o_list, dim=-2)
        
        return ret
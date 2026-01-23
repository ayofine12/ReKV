import torch
import copy
from transformers.models.llama.modeling_llama import LlamaAttention

from eventful_transformer.modules import TokenGate, TokenBuffer
from .kv_cache_manager import ContextManager
from .dot_production_attention import get_multi_stage_dot_production_attention

class EventfulLlamaAttention(LlamaAttention):
    """
    Custom LlamaAttention with eventful QKV gating.
    Reduces computation by only processing changed tokens.
    """
    def __init__(self, config, layer_idx,
                n_init=None, n_local=None, 
                fattn=True, block_size=256,
                topk=8, chunk_size=1,
                max_cached_block=16,
                exc_block_size=256,
                pin_memory=True,
                async_global_stream=None,
    ):
        super().__init__(config, layer_idx)
        
        self.Attn, _ = get_multi_stage_dot_production_attention(fattn)
        self.n_init = n_init
        self.n_local = n_local
        self.fattn = fattn
        self.block_size = block_size
        self.topk = topk
        self.chunk_size = chunk_size
        self.max_cached_block = max_cached_block
        self.exc_block_size = exc_block_size
        self.pin_memory = pin_memory
        self.async_global_stream = async_global_stream

        # Eventful gates and accumulators for QKV
        self.qkv_gate = TokenGate()
        self.qkv_accumulator = TokenBuffer()
        self.projection_gate = TokenGate()
        self.projection_accumulator = TokenBuffer()
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask = None,
        position_bias = None,
        past_key_value = None,
        use_cache = False,
        cache_position = None,
        position_embeddings = None,
        output_attentions = False,
        is_init_prompt = False,
        is_vanilla = False,
        **kwargs,
    ):
        assert not output_attentions

        num_heads = getattr(self, 'num_heads', None)
        if num_heads is None:
            num_heads = self.config.num_attention_heads
        
        num_heads_kv = getattr(self, 'num_key_value_heads', None)
        if num_heads_kv is None:
            num_heads_kv = getattr(self.config, 'num_key_value_heads', num_heads)
        
        dim_head = getattr(self, 'head_dim', None)
        if dim_head is None:
            hidden_size = getattr(self, 'hidden_size', None)
            if hidden_size is None:
                hidden_size = self.config.hidden_size
            dim_head = hidden_size // num_heads

        project_q = self.q_proj
        project_k = self.k_proj
        project_v = self.v_proj
        attention_out = self.o_proj

        batch_size = hidden_states.size(0)
        len_q = hidden_states.size(1)
        len_k = hidden_states.size(1)

        # qkv gate
        if not is_init_prompt and not is_vanilla:
            hidden_states, index = self.qkv_gate(hidden_states)
        query = key_value = hidden_states

        assert use_cache   

        h_q = project_q(query)             # (batch, len_q, num_heads * dim_head)
        h_k = project_k(key_value)         # (batch, len_k, num_heads * dim_head)
        h_v = project_v(key_value)         # (batch, len_k, num_heads * dim_head)

        # qkv accumulator
        if not is_init_prompt and not is_vanilla:
            h_qkv = torch.cat([h_q, h_k, h_v], dim=-1)  # (batch, len, 3 * num_heads * dim_head)
            h_qkv = self.qkv_accumulator(h_qkv, index)
        
            # Split h_qkv back into h_q, h_k, h_v
            hidden_dim = num_heads * dim_head
            h_q = h_qkv[..., :hidden_dim]                           # (batch, len, num_heads * dim_head)
            h_k = h_qkv[..., hidden_dim:2*hidden_dim]               # (batch, len, num_heads * dim_head)
            h_v = h_qkv[..., 2*hidden_dim:]                         # (batch, len, num_heads * dim_head)

        h_q = h_q.view(batch_size, len_q, num_heads, dim_head).permute(0, 2, 1, 3).contiguous()      # (batch, num_heads, len_q, dim_head)
        h_k = h_k.view(batch_size, len_k, num_heads_kv, dim_head).permute(0, 2, 1, 3).contiguous()   # (batch, num_heads_kv, len_k, dim_head)
        h_v = h_v.view(batch_size, len_k, num_heads_kv, dim_head).permute(0, 2, 1, 3).contiguous()   # (batch, num_heads_kv, len_k, dim_head)

        if position_bias._cos_cached is not None and position_bias._cos_cached.device != h_q.device:
            position_bias = copy.deepcopy(position_bias)
            if position_bias.inv_freq.device != h_q.device:
                position_bias.inv_freq = position_bias.inv_freq.to(h_q.device)
            if position_bias._cos_cached is not None:
                position_bias._cos_cached = position_bias._cos_cached.to(h_q.device)
            if position_bias._sin_cached is not None:
                position_bias._sin_cached = position_bias._sin_cached.to(h_q.device)

        if past_key_value is None:
            past_key_value = ContextManager(
                position_bias,
                self.n_init, self.n_local, 
                self.block_size, self.max_cached_block, self.topk, self.chunk_size, self.exc_block_size,
                self.fattn,
                self.async_global_stream,
                self.pin_memory,
            )

        local_q, local_k, local_v = h_q, h_k, h_v
        global_q, global_k, global_v = h_q, h_k, h_v

        # NOTE: Question-answering, fall back to sliding-window attention (infinite_lm)
        if type(past_key_value) is not ContextManager or past_key_value.to_retrieve:
            if type(past_key_value) is ContextManager:  # retrieval
                if past_key_value.retrieved_block_indices is None:  # retrieve based on global_q (question's query)
                    past_k, past_v = past_key_value.get_retrieved_kv(global_q)
                else:  # retrieve based on pre-computed retrieved_block_indices
                    past_k, past_v = past_key_value.get_retrieved_kv()
                updata_kv_cache = False  # We do not update KV cache with the input KV (h_k, h_v) because we only use it for retrieval
            else:  # sliding-window attention
                past_k = past_key_value[0]
                past_v = past_key_value[1]
                updata_kv_cache = True

            """ 2. Update KV w/ past KV cache """
            h_k = torch.cat([past_k, h_k], dim=-2)
            h_v = torch.cat([past_v, h_v], dim=-2)
            len_k += past_k.shape[2]

            """ 3. Update KV cache """
            if updata_kv_cache:
                if len_k <= self.n_local + self.n_init:
                    h_k_cache = h_k
                    h_v_cache = h_v
                else:
                    h_k_cache = torch.cat([h_k[:,:, :self.n_init, :], h_k[:, :, max(0, h_k.size(-2) - self.n_local):, :]], dim=2)
                    h_v_cache = torch.cat([h_v[:,:, :self.n_init, :], h_v[:, :, max(0, h_k.size(-2) - self.n_local):, :]], dim=2)
                current_key_value = (h_k_cache, h_v_cache)
            else:
                current_key_value = (past_k, past_v)

            """ 4. Get local QKV and apply RoPE to local QK """
            h_q_, h_k_, h_v_ = h_q, h_k, h_v
            if len_q + self.n_local < h_k_.size(-2):
                h_k_ = h_k_[:, :, h_k_.size(-2) - len_q - self.n_local:, :]
                h_v_ = h_v_[:, :, h_v_.size(-2) - len_q - self.n_local:, :]

            local_h_q, local_h_k = position_bias(h_q_, h_k_)
            local_h_v = h_v_

            """ 5. Get init QKV and apply RoPE to init Q (Infinite-LM assigns the same position_ids to initial tokens) """
            if len_k > self.n_local:
                init_h_q = position_bias.apply_rotary_pos_emb_one_angle(
                    h_q, self.n_local
                )
                init_h_k = h_k
                init_h_v = h_v
                init_h_k = init_h_k[:, :, :self.n_init, :].contiguous()
                init_h_v = init_h_v[:, :, :self.n_init, :].contiguous()

            else:
                init_h_q = h_q
                init_h_k = torch.empty(
                    (batch_size, num_heads_kv, 0, dim_head),
                    device=h_k.device,
                    dtype=h_k.dtype
                )
                init_h_v = torch.empty(
                    (batch_size, num_heads_kv, 0, dim_head),
                    device=h_v.device,
                    dtype=h_v.dtype
                )

            """ 6. Sliding Window Attention """
            attn = self.Attn(local_h_q.shape, local_h_q.dtype, local_h_q.device)
            attn.append(local_h_q, local_h_k, local_h_v, sliding_window=self.n_local)
            attn.append(init_h_q, init_h_k, init_h_v, end=True, sliding_window=(len_k - len_q, self.n_local), complement_sliding_window=True)
            score, _ = attn.get_result()

            score = score.view(batch_size, num_heads, len_q, dim_head).permute(0, 2, 1, 3) # (batch, len_q, num_heads, dim_head)
            score = score.reshape(batch_size, len_q, num_heads * dim_head) # (batch, len_q, num_heads * dim_head)
            score = attention_out(score)

            return score, current_key_value

        # NOTE: Encode video, managed by the KVCacheManager
        else:
            o = past_key_value.append(
                local_q, local_k, local_v,
                global_q, global_k, global_v,
            )
            o = o.view(batch_size, num_heads, len_q, dim_head).permute(0, 2, 1, 3)
            o = o.reshape(batch_size, len_q, dim_head * num_heads)

            # projection gate
            if not is_init_prompt and not is_vanilla:
                o, index = self.projection_gate(o)  
            o = attention_out(o)
            # projection accumulator
            if not is_init_prompt and not is_vanilla:
                o = self.projection_accumulator(o, index)

            return o, past_key_value

    def reset_self(self):
        """Reset for new sequence."""
        self.first = True
        self.qkv_gate.reset_self()
        self.qkv_accumulator.reset_self()
        self.projection_gate.reset_self()
        self.projection_accumulator.reset_self()



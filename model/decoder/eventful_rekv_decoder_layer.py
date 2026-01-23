import torch
import copy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from eventful_transformer.modules import TokenGate, TokenBuffer


class EventfulLlamaDecoderLayer(LlamaDecoderLayer):
    """
    EventfulLlamaDecoderLayer extends LlamaDecoderLayer with custom forward logic.
    This implementation is based on decoder_layer_forward from patch.py (lines 160-208).
    """

    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.mlp_gate = TokenGate()
        self.mlp_accumulator = TokenBuffer()
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,  # Support both old and new parameter names
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        output_attentions=False,
        is_vanilla=False,
        **kwargs,
    ):
        """
        Forward pass for EventfulLlamaDecoderLayer.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask tensor
            position_ids: Position IDs tensor
            past_key_value: Cached key/value states for fast decoding
            use_cache: Whether to return cached key/value states
            cache_position: Position in the cache
            position_embeddings: Pre-computed position embeddings
            output_attentions: Whether to output attention weights
            **kwargs: Additional keyword arguments
            
        Returns:
            Tuple of outputs depending on use_cache and output_attentions:
            - output_attentions=False, use_cache=True: (hidden_states, past_key_value)
            - output_attentions=True, use_cache=True: (hidden_states, attn_weights, past_key_value)
            - use_cache=False: (hidden_states,) or (hidden_states, attn_weights)
        """
        
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self Attention - unpack 3 values: (attn_output, attn_weights, past_key_value)
        hidden_states, past_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_bias=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
            is_vanilla=is_vanilla,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if not is_vanilla:
            hidden_states, index = self.mlp_gate(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if not is_vanilla:
            hidden_states = self.mlp_accumulator(hidden_states, index)
        hidden_states = residual + hidden_states
        
        # Return format to match model_forward expectations:
        # - output_attentions=False, use_cache=True: (hidden_states, past_key_value)
        # - output_attentions=True, use_cache=True: (hidden_states, attn_weights, past_key_value)
        # - use_cache=False: (hidden_states,) or (hidden_states, attn_weights)
        if use_cache:
            if output_attentions:
                return hidden_states, None, past_key_value
            else:
                return hidden_states, past_key_value
        else:
            if output_attentions:
                return hidden_states, None
            else:
                return hidden_states

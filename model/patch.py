import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
from transformers.models.llama.modeling_llama import LlamaModel
from transformers.models.mistral.modeling_mistral import MistralModel
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model as Qwen2BaseModel

from model.attention import RotaryEmbeddingESM, rekv_attention_forward


def huggingface_forward(forward):
    def hf_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask = None,
        position_ids = None,
        past_key_value = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        assert not output_attentions
        # Safely access attributes, falling back to config if needed
        num_heads = getattr(self, 'num_heads', None)
        if num_heads is None:
            num_heads = self.config.num_attention_heads
        
        num_key_value_heads = getattr(self, 'num_key_value_heads', None)
        if num_key_value_heads is None:
            num_key_value_heads = getattr(self.config, 'num_key_value_heads', num_heads)
        
        head_dim = getattr(self, 'head_dim', None)
        if head_dim is None:
            hidden_size = getattr(self, 'hidden_size', None)
            if hidden_size is None:
                hidden_size = self.config.hidden_size
            head_dim = hidden_size // num_heads
        
        ret = forward(
            self, hidden_states, hidden_states,
            position_ids, use_cache, past_key_value,
            self.q_proj, self.k_proj, self.v_proj, self.o_proj, 
            head_dim, num_heads, num_key_value_heads
        )
        if use_cache:
            o, pkv = ret
        else:
            o = ret
            pkv = None

        return o, None, pkv

    return hf_forward


def patch_hf(
    model,
    attn_kwargs: dict = {},
    base = None, 
    distance_scale = None,
    **kwargs
):
    attn_kwargs.update(kwargs)
    # This approach lacks scalability and will be refactored.
    from transformers import LlamaForCausalLM, MistralForCausalLM, Qwen2ForCausalLM, Qwen2Model
    from transformers.models.llama.modeling_llama import LlamaAttention, LlamaModel, BaseModelOutputWithPast

    def model_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask = None,
        position_ids = None,
        past_key_values = None,
        inputs_embeds = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
        *args,
        **kwargs
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            if hasattr(self, "config") and hasattr(self.config, "scale_emb"):
                inputs_embeds = inputs_embeds * self.config.scale_emb

        if use_cache:
            pkv = tuple()

        else:
            pkv = None

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for i, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=self.position_bias,
                past_key_value=past_key_values[i] if past_key_values is not None else None,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                _cache = layer_outputs[2 if output_attentions else 1]
                pkv = pkv + (_cache,)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, pkv, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=pkv,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    forward = huggingface_forward(rekv_attention_forward(**attn_kwargs))
    
    # Patch LlamaDecoderLayer.forward to handle 3 return values from self_attn
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
    
    def decoder_layer_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask = None,
        position_ids = None,
        past_key_value = None,  # Support both old and new parameter names
        use_cache = False,
        cache_position = None,
        position_embeddings = None,
        output_attentions = False,
        **kwargs,
    ):
        
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention - unpack 3 values: (attn_output, attn_weights, past_key_value)
        hidden_states, _, past_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
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
    
    if isinstance(model, LlamaForCausalLM):
        Attention = model.model.layers[0].self_attn.__class__
        Model = model.model.__class__
        base_model = model.model
    elif isinstance(model, LlamaModel):
        # LlamaModel directly has layers attribute (not model.model.layers)
        Attention = model.layers[0].self_attn.__class__
        Model = model.__class__
        base_model = model
    elif isinstance(model, MistralForCausalLM):
        Attention = model.model.layers[0].self_attn.__class__
        Model = model.model.__class__
        base_model = model.model
    elif isinstance(model, MistralModel):
        Attention = model.layers[0].self_attn.__class__
        Model = model.__class__
        base_model = model
    elif isinstance(model, Qwen2ForCausalLM):
        Attention = model.model.layers[0].self_attn.__class__
        Model = model.model.__class__
        base_model = model.model
    elif isinstance(model, Qwen2Model):
        Attention = model.model.layers[0].self_attn.__class__
        Model = model.model.__class__
        base_model = model.model
    elif isinstance(model, Qwen2BaseModel):
        Attention = model.layers[0].self_attn.__class__
        Model = model.__class__
        base_model = model
    elif model.__class__.__name__ == "MiniCPMForCausalLM":
        Attention = model.model.layers[0].self_attn.__class__
        Model = model.model.__class__
        base_model = model.model
    else:
        raise ValueError(f"Only supports llama, mistral and qwen2 models, not {model.__class__.__name__}.")

    # In newer transformers versions, rotary_emb might not be an attribute of attention
    # We get RoPE parameters from config instead
    attention = base_model.layers[0].self_attn
    config = attention.config
    
    if hasattr(attention, 'rotary_emb'):
        # Old transformers: rotary_emb exists
        hf_rope = attention.rotary_emb
        if isinstance(hf_rope, Qwen2RotaryEmbedding):
            base = hf_rope.base
            distance_scale = 1.0
            dim = hf_rope.dim
        else:
            base = hf_rope.config.rope_theta
            distance_scale = distance_scale if distance_scale is not None else 1.0
            partial_rotary_factor = hf_rope.config.partial_rotary_factor if hasattr(hf_rope.config, "partial_rotary_factor") else 1.0
            dim = int((hf_rope.config.hidden_size // hf_rope.config.num_attention_heads) * partial_rotary_factor)
    else:
        # New transformers: get RoPE parameters from config
        base = getattr(config, 'rope_theta', 10000.0)
        distance_scale = distance_scale if distance_scale is not None else 1.0
        partial_rotary_factor = getattr(config, 'partial_rotary_factor', 1.0)
        
        # Calculate dim based on head_dim or hidden_size
        if hasattr(attention, 'head_dim'):
            dim = int(attention.head_dim * partial_rotary_factor)
        else:
            dim = int((config.hidden_size // config.num_attention_heads) * partial_rotary_factor)
    rope = RotaryEmbeddingESM(
        dim,
        base,
        distance_scale
    )
    base_model.position_bias = rope

    def set_forward(m): # m is module object
        if isinstance(m, Attention):
            m._old_forward = m.forward
            m.forward = forward.__get__(m, Attention)
        elif isinstance(m, (LlamaDecoderLayer, MistralDecoderLayer, Qwen2DecoderLayer)):
            m._old_forward = m.forward
            m.forward = decoder_layer_forward.__get__(m, m.__class__)

    model.apply(set_forward)

    base_model._old_forward = base_model.forward
    base_model.forward = model_forward.__get__(base_model, Model)

    return model

import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
from transformers.models.llama.modeling_llama import LlamaModel
from transformers.models.mistral.modeling_mistral import MistralModel
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model as Qwen2BaseModel

from model.attention import RotaryEmbeddingESM, EventfulLlamaAttention
from model.decoder import EventfulLlamaDecoderLayer


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
        is_vanilla=False,
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
                is_vanilla=is_vanilla,
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

    # forward = huggingface_forward(rekv_attention_forward(**attn_kwargs))
    
    # Get the appropriate model class and base model
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
    
    print(f"Replacing {len(base_model.layers)} decoder layers with EventfulLlamaDecoderLayer...")
    new_layers = []
    for i, old_layer in enumerate(base_model.layers):
        # Get device from old layer
        device = old_layer.self_attn.q_proj.weight.device
        
        # Create new EventfulLlamaDecoderLayer
        new_layer = EventfulLlamaDecoderLayer(config, i)
        
        # Move to device BEFORE loading state_dict to ensure all new components are on correct device
        new_layer = new_layer.to(device)
        
        # Copy all weights from old layer to new layer
        # This includes self_attn, mlp, input_layernorm, post_attention_layernorm
        new_layer.load_state_dict(old_layer.state_dict(), strict=False)
        
        # Now replace the attention layer with EventfulLlamaAttention
        old_attention = new_layer.self_attn
        new_attention = EventfulLlamaAttention(config, i, **attn_kwargs)
        
        # Move attention to device BEFORE loading state_dict
        new_attention = new_attention.to(device)
        
        # Copy weights from old attention to new attention
        new_attention.load_state_dict(old_attention.state_dict(), strict=False)
        
        # Replace attention in the new layer
        new_layer.self_attn = new_attention
        
        # Disable gradient computation for inference (memory optimization)
        # Note: This is in addition to model.eval() and @torch.inference_mode()
        # to ensure stateful modules (TokenGate, TokenBuffer) don't hold gradient graphs
        new_layer.requires_grad_(False)
        
        new_layers.append(new_layer)
    
    # Replace all layers at once
    base_model.layers = torch.nn.ModuleList(new_layers)

    base_model._old_forward = base_model.forward
    base_model.forward = model_forward.__get__(base_model, Model)

    return model
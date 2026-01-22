"""
Utility functions for ReKV models.
"""
from model.eventful_transformer.modules import SimpleSTGTGate, TokenDeltaGate, TokenGate


def set_policies(model, policy_class, **policy_kwargs):
    """
    Set the policy for all gates in the model.
    
    Args:
        model: The model containing gates (should have modules() method)
        policy_class: The policy class to instantiate (e.g., TokenNormTopK)
        **policy_kwargs: Keyword arguments to pass to the policy constructor
    
    Example:
        from model.eventful_transformer.policies import TokenNormTopK
        from model.utils import set_policies
        
        # Set policy to select top 128 tokens
        set_policies(model, TokenNormTopK, k=128)
    """
    # Find base model (decoder layers)
    # Try multiple paths to find the actual decoder layers
    if hasattr(model, 'language_model'):
        language_model = model.language_model
        if hasattr(language_model, 'model'):
            base_model = language_model.model
        else:
            base_model = language_model
    elif hasattr(model, 'model'):
        # LlamaForCausalLM case
        base_model = model.model
        model_type = "LlamaForCausalLM"
    else:
        # LlamaModel case
        base_model = model
        model_type = "LlamaModel"
    
    # Iterate through decoder layers
    for _, layer in enumerate(base_model.layers):
        # Check if layer has self_attn
        if not hasattr(layer, 'self_attn'):
            continue
        
        attention = layer.self_attn
        
        # Check direct attributes (for EventfulLlamaAttention)
        gate_attr_names = ['qkv_gate', 'projection_gate', 'mlp_gate', 'v_gate', 'matmul_gate']
        
        for attr_name in gate_attr_names:
            if hasattr(attention, attr_name):
                gate = getattr(attention, attr_name)
                gate_type_name = type(gate).__name__
                
                # Check by type name (handles import path issues)
                is_gate = gate_type_name in ['SimpleSTGTGate', 'TokenDeltaGate', 'TokenGate']
                
                # Also check if it has policy attribute (more reliable)
                has_policy_attr = hasattr(gate, 'policy')
                
                if is_gate or has_policy_attr:
                    # Set policy
                    if has_policy_attr:
                        gate.policy = policy_class(**policy_kwargs)
    else:
        print(f"\n⚠ WARNING: No gates found in the model!")
        print("  This model may not have EventfulTransformer modules.")
        print("=" * 70 + "\n")


def count_gates(model):
    """
    Count the number of gates in the model.
    
    Args:
        model: The model containing gates
    
    Returns:
        dict: Dictionary with counts for each gate type
    """
    gate_classes = [SimpleSTGTGate, TokenDeltaGate, TokenGate]
    counts = {}
    
    for gate_class in gate_classes:
        count = sum(1 for module in model.modules() if isinstance(module, gate_class))
        counts[gate_class.__name__] = count
    
    return counts


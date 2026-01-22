"""
mlp_gate에 policy가 제대로 설정되었는지 확인하는 스크립트
"""

def verify_mlp_gate_policy(model, expected_policy_class):
    """
    모든 layer의 mlp_gate에 policy가 설정되었는지 확인
    """
    print("=" * 70)
    print("Verifying MLP Gate Policies")
    print("=" * 70)
    
    # Find base model
    if hasattr(model, 'language_model'):
        base_model = model.language_model
    elif hasattr(model, 'model'):
        base_model = model.model
    else:
        base_model = model
    
    print(f"Base model type: {type(base_model).__name__}")
    print(f"Total layers: {len(base_model.layers)}")
    
    # Check each layer
    layers_with_mlp_gate = 0
    layers_with_policy = 0
    
    for layer_idx, layer in enumerate(base_model.layers):
        # Check if layer has mlp_gate
        if hasattr(layer, 'mlp_gate'):
            layers_with_mlp_gate += 1
            mlp_gate = layer.mlp_gate
            
            # Check if policy is set
            if hasattr(mlp_gate, 'policy') and mlp_gate.policy is not None:
                policy_type = type(mlp_gate.policy).__name__
                expected_type = expected_policy_class.__name__
                
                if policy_type == expected_type:
                    layers_with_policy += 1
                    if layer_idx < 3 or layer_idx >= len(base_model.layers) - 3:
                        print(f"  Layer {layer_idx}: mlp_gate.policy = {policy_type} ✓")
                elif layer_idx == 3:
                    print(f"  ...")
            else:
                if layer_idx < 3:
                    print(f"  Layer {layer_idx}: mlp_gate.policy = None ✗")
    
    print(f"\nSummary:")
    print(f"  Layers with mlp_gate: {layers_with_mlp_gate}/{len(base_model.layers)}")
    print(f"  Layers with policy set: {layers_with_policy}/{layers_with_mlp_gate}")
    
    # Also check attention gates
    print(f"\nChecking attention gates (qkv_gate, projection_gate):")
    attn_gates_with_policy = 0
    total_attn_gates = 0
    
    for layer_idx, layer in enumerate(base_model.layers):
        if hasattr(layer, 'self_attn'):
            attention = layer.self_attn
            for gate_name in ['qkv_gate', 'projection_gate']:
                if hasattr(attention, gate_name):
                    total_attn_gates += 1
                    gate = getattr(attention, gate_name)
                    if hasattr(gate, 'policy') and gate.policy is not None:
                        attn_gates_with_policy += 1
    
    print(f"  Attention gates with policy: {attn_gates_with_policy}/{total_attn_gates}")
    
    # Final verdict
    print("\n" + "=" * 70)
    if layers_with_policy == layers_with_mlp_gate and attn_gates_with_policy == total_attn_gates:
        print("✓ SUCCESS: All gates have policies set!")
    else:
        print("✗ WARNING: Some gates are missing policies!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    print("""
사용 방법:

from model.video_llava_rekv import load_model
from model.utils import set_policies
from model.eventful_transformer.policies import TokenNormTopK
from verify_mlp_gate import verify_mlp_gate_policy

# 모델 로드
model, processor = load_model(
    model_path='/mnt/models/Video-LLaVA-7B-hf',
    n_local=15000,
    topk=64,
    chunk_size=1
)

# Policy 설정
set_policies(model, TokenNormTopK, k=128)

# 검증
verify_mlp_gate_policy(model, TokenNormTopK)
    """)


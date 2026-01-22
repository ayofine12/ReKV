"""
빠르고 간단한 patch 검증 스크립트
실제 사용 예시 포함
"""
import torch
from model.decoder import EventfulLlamaDecoderLayer
from model.attention import EventfulLlamaAttention


def quick_verify(model, verbose=True):
    """
    Patch가 제대로 적용되었는지 빠르게 확인
    
    Args:
        model: Patched 모델
        verbose: 상세 출력 여부
    
    Returns:
        bool: 검증 통과 여부
    """
    # Base model 찾기
    if hasattr(model, 'language_model'):
        base_model = model.language_model.model
        model_type = "VideoLlava/LongVA"
    elif hasattr(model, 'model'):
        base_model = model.model
        model_type = "LlamaForCausalLM"
    else:
        base_model = model
        model_type = "LlamaModel"
    
    num_layers = len(base_model.layers)
    
    # 1. Layer 타입 확인
    eventful_layers = sum(1 for layer in base_model.layers 
                         if isinstance(layer, EventfulLlamaDecoderLayer))
    
    # 2. Attention 타입 확인
    eventful_attentions = sum(1 for layer in base_model.layers 
                             if isinstance(layer.self_attn, EventfulLlamaAttention))
    
    # 3. Position bias 확인
    has_position_bias = hasattr(base_model, 'position_bias')
    
    # 결과 출력
    if verbose:
        print("=" * 60)
        print("Quick Patch Verification")
        print("=" * 60)
        print(f"Model type: {model_type}")
        print(f"Total layers: {num_layers}")
        print(f"EventfulLlamaDecoderLayer: {eventful_layers}/{num_layers} {'✓' if eventful_layers == num_layers else '✗'}")
        print(f"EventfulLlamaAttention: {eventful_attentions}/{num_layers} {'✓' if eventful_attentions == num_layers else '✗'}")
        print(f"Position bias: {'✓' if has_position_bias else '✗'}")
        
        # 첫 번째와 마지막 레이어 타입 출력
        print(f"\nFirst layer: {type(base_model.layers[0]).__name__}")
        print(f"Last layer: {type(base_model.layers[-1]).__name__}")
        print(f"First attention: {type(base_model.layers[0].self_attn).__name__}")
        print("=" * 60)
    
    # 모든 검사 통과 여부
    all_passed = (
        eventful_layers == num_layers and 
        eventful_attentions == num_layers and 
        has_position_bias
    )
    
    if verbose:
        if all_passed:
            print("✓ All checks passed! Patch applied successfully.")
        else:
            print("✗ Some checks failed! Patch may not be properly applied.")
        print("=" * 60)
    
    return all_passed


def print_layer_structure(model, layer_idx=0):
    """
    특정 레이어의 구조를 자세히 출력
    
    Args:
        model: 모델
        layer_idx: 확인할 레이어 인덱스
    """
    if hasattr(model, 'language_model'):
        base_model = model.language_model.model
    elif hasattr(model, 'model'):
        base_model = model.model
    else:
        base_model = model
    
    layer = base_model.layers[layer_idx]
    
    print(f"\n{'=' * 60}")
    print(f"Layer {layer_idx} Structure")
    print(f"{'=' * 60}")
    print(f"Layer type: {type(layer).__name__}")
    print(f"Layer class module: {type(layer).__module__}")
    print(f"\nAttributes:")
    for attr_name in dir(layer):
        if not attr_name.startswith('_'):
            attr = getattr(layer, attr_name)
            if not callable(attr):
                print(f"  - {attr_name}: {type(attr).__name__}")
    
    print(f"\nAttention type: {type(layer.self_attn).__name__}")
    print(f"Attention class module: {type(layer.self_attn).__module__}")
    
    # Attention의 주요 속성 확인
    attn = layer.self_attn
    print(f"\nAttention attributes:")
    if hasattr(attn, 'layer_idx'):
        print(f"  - layer_idx: {attn.layer_idx}")
    if hasattr(attn, 'q_proj'):
        print(f"  - q_proj: {attn.q_proj.weight.shape}")
    if hasattr(attn, 'k_proj'):
        print(f"  - k_proj: {attn.k_proj.weight.shape}")
    if hasattr(attn, 'v_proj'):
        print(f"  - v_proj: {attn.v_proj.weight.shape}")
    if hasattr(attn, 'o_proj'):
        print(f"  - o_proj: {attn.o_proj.weight.shape}")
    
    print(f"{'=' * 60}")


# 사용 예시
if __name__ == "__main__":
    print("""
사용 예시:

# 1. 모델 로드 및 patch 적용
from model.video_llava_rekv import load_model
model, processor = load_model(
    model_path='/mnt/models/Video-LLaVA-7B-hf',
    n_local=15000,
    topk=64,
    chunk_size=1
)

# 2. 빠른 검증
from quick_verify import quick_verify, print_layer_structure

# 간단한 검증
result = quick_verify(model)

# 특정 레이어 자세히 보기
print_layer_structure(model, layer_idx=0)
print_layer_structure(model, layer_idx=-1)  # 마지막 레이어

# 3. Python/IPython에서 직접 확인
from model.decoder import EventfulLlamaDecoderLayer
base_model = model.language_model.model
print(type(base_model.layers[0]))
print(isinstance(base_model.layers[0], EventfulLlamaDecoderLayer))
    """)


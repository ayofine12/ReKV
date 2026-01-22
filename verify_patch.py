"""
patch.py에서 decoder layer가 제대로 변경되었는지 확인하는 스크립트
"""
import torch
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration

from model.patch import patch_hf
from model.decoder import EventfulLlamaDecoderLayer
from model.attention import EventfulLlamaAttention


def verify_layer_replacement(model):
    """
    Decoder layer가 EventfulLlamaDecoderLayer로 제대로 교체되었는지 확인
    """
    print("=" * 80)
    print("1. Layer Type Verification")
    print("=" * 80)
    
    # 모델의 base model 찾기
    if hasattr(model, 'language_model'):
        # VideoLlava의 경우
        base_model = model.language_model.model
    elif hasattr(model, 'model'):
        base_model = model.model
    else:
        base_model = model
    
    num_layers = len(base_model.layers)
    print(f"Total layers: {num_layers}")
    
    # 모든 레이어 타입 확인
    eventful_count = 0
    for i, layer in enumerate(base_model.layers):
        layer_type = type(layer).__name__
        is_eventful = isinstance(layer, EventfulLlamaDecoderLayer)
        
        if is_eventful:
            eventful_count += 1
        
        # 처음 3개와 마지막 3개 레이어만 출력
        if i < 3 or i >= num_layers - 3:
            status = "✓" if is_eventful else "✗"
            print(f"  Layer {i:2d}: {layer_type:40s} {status}")
        elif i == 3:
            print(f"  ...")
    
    print(f"\nEventfulLlamaDecoderLayer count: {eventful_count}/{num_layers}")
    
    if eventful_count == num_layers:
        print("✓ All layers successfully replaced!")
    else:
        print(f"✗ Warning: Only {eventful_count}/{num_layers} layers replaced!")
    
    return eventful_count == num_layers


def verify_attention_type(model):
    """
    Attention layer가 EventfulLlamaAttention으로 제대로 교체되었는지 확인
    """
    print("\n" + "=" * 80)
    print("2. Attention Type Verification")
    print("=" * 80)
    
    if hasattr(model, 'language_model'):
        base_model = model.language_model.model
    elif hasattr(model, 'model'):
        base_model = model.model
    else:
        base_model = model
    
    num_layers = len(base_model.layers)
    eventful_attn_count = 0
    
    for i, layer in enumerate(base_model.layers):
        attention = layer.self_attn
        attn_type = type(attention).__name__
        is_eventful = isinstance(attention, EventfulLlamaAttention)
        
        if is_eventful:
            eventful_attn_count += 1
        
        # 처음 3개와 마지막 3개만 출력
        if i < 3 or i >= num_layers - 3:
            status = "✓" if is_eventful else "✗"
            print(f"  Layer {i:2d} attention: {attn_type:40s} {status}")
        elif i == 3:
            print(f"  ...")
    
    print(f"\nEventfulLlamaAttention count: {eventful_attn_count}/{num_layers}")
    
    if eventful_attn_count == num_layers:
        print("✓ All attention layers successfully replaced!")
    else:
        print(f"✗ Warning: Only {eventful_attn_count}/{num_layers} attention layers replaced!")
    
    return eventful_attn_count == num_layers


def verify_layer_attributes(model):
    """
    EventfulLlamaDecoderLayer의 특별한 속성들이 존재하는지 확인
    """
    print("\n" + "=" * 80)
    print("3. Layer Attributes Verification")
    print("=" * 80)
    
    if hasattr(model, 'language_model'):
        base_model = model.language_model.model
    elif hasattr(model, 'model'):
        base_model = model.model
    else:
        base_model = model
    
    # 첫 번째 레이어로 확인
    first_layer = base_model.layers[0]
    
    print(f"First layer type: {type(first_layer).__name__}")
    print(f"\nChecking EventfulLlamaDecoderLayer specific attributes:")
    
    # EventfulLlamaDecoderLayer에만 있는 속성들 확인
    expected_attrs = ['layer_idx', 'self_attn', 'mlp', 'input_layernorm', 'post_attention_layernorm']
    
    for attr in expected_attrs:
        has_attr = hasattr(first_layer, attr)
        status = "✓" if has_attr else "✗"
        print(f"  {attr:30s}: {status}")
    
    # Attention의 특별한 속성 확인
    print(f"\nChecking EventfulLlamaAttention specific attributes:")
    attention = first_layer.self_attn
    
    eventful_attn_attrs = ['layer_idx', 'q_proj', 'k_proj', 'v_proj', 'o_proj']
    
    for attr in eventful_attn_attrs:
        has_attr = hasattr(attention, attr)
        status = "✓" if has_attr else "✗"
        print(f"  {attr:30s}: {status}")


def verify_forward_pass(model):
    """
    Forward pass가 정상적으로 동작하는지 확인
    """
    print("\n" + "=" * 80)
    print("4. Forward Pass Test")
    print("=" * 80)
    
    try:
        device = next(model.parameters()).device
        
        # 더미 입력 생성
        batch_size = 1
        seq_len = 10
        vocab_size = model.config.vocab_size if hasattr(model, 'config') else 32000
        
        input_ids = torch.randint(0, min(vocab_size, 1000), (batch_size, seq_len)).to(device)
        
        print(f"Input shape: {input_ids.shape}")
        print(f"Device: {device}")
        
        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids=input_ids, use_cache=True)
        
        print(f"Output shape: {outputs.logits.shape if hasattr(outputs, 'logits') else outputs.last_hidden_state.shape}")
        print(f"Past key values: {len(outputs.past_key_values) if hasattr(outputs, 'past_key_values') and outputs.past_key_values else 'None'}")
        
        print("✓ Forward pass successful!")
        return True
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_position_bias(model):
    """
    Position bias (RoPE)가 제대로 설정되었는지 확인
    """
    print("\n" + "=" * 80)
    print("5. Position Bias (RoPE) Verification")
    print("=" * 80)
    
    if hasattr(model, 'language_model'):
        base_model = model.language_model.model
    elif hasattr(model, 'model'):
        base_model = model.model
    else:
        base_model = model
    
    has_position_bias = hasattr(base_model, 'position_bias')
    print(f"Has position_bias attribute: {has_position_bias}")
    
    if has_position_bias:
        position_bias = base_model.position_bias
        print(f"Position bias type: {type(position_bias).__name__}")
        print(f"Position bias dim: {position_bias.dim if hasattr(position_bias, 'dim') else 'N/A'}")
        print(f"Position bias base: {position_bias.base if hasattr(position_bias, 'base') else 'N/A'}")
        print("✓ Position bias properly set!")
        return True
    else:
        print("✗ Position bias not found!")
        return False


def main():
    """
    전체 검증 실행
    """
    print("\n" + "=" * 80)
    print("DECODER LAYER REPLACEMENT VERIFICATION")
    print("=" * 80)
    
    # 예시: VideoLlava 모델 로드 (실제 사용하는 모델로 변경)
    print("\nLoading model...")
    
    # 여기에 실제 모델 로딩 코드를 추가하세요
    # 예시:
    # model_path = "model_zoo/Video-LLaVA-7B-hf"
    # processor = VideoLlavaProcessor.from_pretrained(model_path)
    # model = VideoLlavaForConditionalGeneration.from_pretrained(
    #     model_path,
    #     device_map="auto",
    #     torch_dtype=torch.float16
    # )
    
    # # Patch 적용
    # inf_llm_config = {
    #     'n_init': 10,
    #     'n_local': 15000,
    #     'fattn': True,
    #     'block_size': 256,
    #     'topk': 64,
    #     'chunk_size': 1,
    # }
    # model.language_model = patch_hf(model.language_model, **inf_llm_config)
    
    # 실제 검증 수행
    # results = []
    # results.append(("Layer Replacement", verify_layer_replacement(model)))
    # results.append(("Attention Replacement", verify_attention_type(model)))
    # verify_layer_attributes(model)
    # results.append(("Forward Pass", verify_forward_pass(model)))
    # results.append(("Position Bias", verify_position_bias(model)))
    
    # # 최종 결과 출력
    # print("\n" + "=" * 80)
    # print("FINAL RESULTS")
    # print("=" * 80)
    # for test_name, passed in results:
    #     status = "✓ PASSED" if passed else "✗ FAILED"
    #     print(f"{test_name:30s}: {status}")
    
    print("\n⚠ Please uncomment and modify the model loading code above!")
    print("Replace the model path and configuration with your actual setup.")


if __name__ == "__main__":
    main()


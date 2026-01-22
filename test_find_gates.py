"""
TokenGate를 찾는 여러 방법 테스트
"""
from model.eventful_transformer.modules import TokenGate, TokenBuffer

def test_find_gates(model):
    """
    모델에서 TokenGate를 찾는 여러 방법을 테스트
    """
    print("=" * 70)
    print("Testing Gate Detection Methods")
    print("=" * 70)
    
    # Find base model
    if hasattr(model, 'language_model'):
        print(f"Model has 'language_model' attribute")
        language_model = model.language_model
        print(f"  language_model type: {type(language_model).__name__}")
        
        # Check if language_model has .model
        if hasattr(language_model, 'model'):
            print(f"  language_model.model type: {type(language_model.model).__name__}")
            base_model = language_model.model
        else:
            print(f"  language_model has no .model attribute")
            base_model = language_model
    elif hasattr(model, 'model'):
        base_model = model.model
    else:
        base_model = model
    
    print(f"\nUsing base_model type: {type(base_model).__name__}")
    
    # Check if it has layers
    if not hasattr(base_model, 'layers'):
        print(f"❌ base_model has no 'layers' attribute!")
        print(f"Available attributes: {[a for a in dir(base_model) if not a.startswith('_')][:30]}")
        return
    
    print(f"✓ base_model has {len(base_model.layers)} layers")
    
    # Check first layer
    first_layer = base_model.layers[0]
    print(f"\nFirst layer type: {type(first_layer).__name__}")
    print(f"First layer module: {type(first_layer).__module__}")
    
    # Check if it has self_attn
    if not hasattr(first_layer, 'self_attn'):
        print(f"❌ First layer has no 'self_attn' attribute!")
        return
    
    attention = first_layer.self_attn
    print(f"\nFirst attention type: {type(attention).__name__}")
    print(f"First attention module: {type(attention).__module__}")
    
    # Method 1: Check direct attributes
    print(f"\n--- Method 1: Direct Attribute Access ---")
    for attr_name in ['qkv_gate', 'qkv_accumulator', 'projection_gate', 'projection_accumulator']:
        if hasattr(attention, attr_name):
            obj = getattr(attention, attr_name)
            print(f"✓ Found {attr_name}: {type(obj).__name__}")
            print(f"  Is TokenGate? {isinstance(obj, TokenGate)}")
            print(f"  Is TokenBuffer? {isinstance(obj, TokenBuffer)}")
            print(f"  Has policy? {hasattr(obj, 'policy')}")
            if hasattr(obj, 'policy'):
                print(f"  Policy value: {obj.policy}")
        else:
            print(f"❌ No {attr_name}")
    
    # Method 2: named_modules()
    print(f"\n--- Method 2: named_modules() ---")
    gates_found = []
    for name, module in attention.named_modules():
        if isinstance(module, (TokenGate, TokenBuffer)):
            gates_found.append((name, type(module).__name__))
    
    if gates_found:
        print(f"Found {len(gates_found)} gates/buffers:")
        for name, type_name in gates_found:
            print(f"  - {name}: {type_name}")
    else:
        print(f"❌ No gates found via named_modules()")
    
    # Method 3: modules()
    print(f"\n--- Method 3: modules() ---")
    gate_count = 0
    for module in attention.modules():
        if isinstance(module, (TokenGate, TokenBuffer)):
            gate_count += 1
    print(f"Found {gate_count} gates/buffers via modules()")
    
    # Method 4: Check _modules dict
    print(f"\n--- Method 4: _modules dictionary ---")
    if hasattr(attention, '_modules'):
        print(f"Attention._modules keys: {list(attention._modules.keys())}")
        for key in attention._modules.keys():
            module = attention._modules[key]
            if isinstance(module, (TokenGate, TokenBuffer)):
                print(f"  ✓ {key}: {type(module).__name__}")
    
    # Method 5: All attributes
    print(f"\n--- Method 5: All Attributes ---")
    all_attrs = [a for a in dir(attention) if not a.startswith('_')]
    gate_attrs = [a for a in all_attrs if 'gate' in a.lower() or 'buffer' in a.lower() or 'accumulator' in a.lower()]
    print(f"Potential gate attributes: {gate_attrs}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    print("""
사용 방법:

from model.video_llava_rekv import load_model
from test_find_gates import test_find_gates

model, processor = load_model(
    model_path='/mnt/models/Video-LLaVA-7B-hf',
    n_local=15000,
    topk=64,
    chunk_size=1
)

test_find_gates(model)
    """)


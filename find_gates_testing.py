from model.video_llava_rekv import load_model
from test_find_gates import test_find_gates

model, processor = load_model(
    model_path='/mnt/models/Video-LLaVA-7B-hf',
    n_local=15000,
    topk=64,
    chunk_size=1
)

test_find_gates(model)

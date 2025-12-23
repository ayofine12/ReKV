import torch
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration
from logzero import logger

from model.patch import patch_hf
from model.abstract_rekv import Abstract_ReKV


class VideoLlava_ReKV(VideoLlavaForConditionalGeneration, Abstract_ReKV):
    def __init__(self, config, processor, n_frame_tokens, init_prompt_ids, n_local, topk, chunk_size):
        VideoLlavaForConditionalGeneration.__init__(self, config)
        Abstract_ReKV.__init__(self, processor, n_frame_tokens, init_prompt_ids, n_local, topk, chunk_size)
        self.processor.video_processor = self.processor.image_processor

    def get_prompt(self, query, mc=False):
        prompt =  f"\n{query} ASSISTANT:"
        if mc:
            prompt += ' Best option: ('
        return prompt

    def get_kv_cache_info(self):
        """KV 캐시 상태 정보를 반환합니다.
        
        Returns:
            dict: KV 캐시 상태 정보
                - total_tokens: 전체 토큰 수 (모든 레이어의 평균)
                - num_layers: 레이어 수
                - local_kv_size: Local KV 캐시 크기 (토큰 수)
                - global_remainder_size: Global remainder 크기 (토큰 수)
                - num_global_blocks: CPU에 저장된 global block 수
                - init_exc: Init KV 캐시가 가득 찼는지 여부
                - cpu_memory_gb: CPU 메모리 사용량 (GB)
        """
        if self.kv_cache is None:
            return {
                'total_tokens': 0,
                'num_layers': 0,
                'local_kv_size': 0,
                'global_remainder_size': 0,
                'num_global_blocks': 0,
                'init_exc': False,
                'cpu_memory_gb': 0.0
            }
        
        # 첫 번째 레이어의 ContextManager를 사용하여 정보 수집
        first_layer = self.kv_cache[0]

        if hasattr(first_layer, 'size'):
            global_kv_cache_size = first_layer.size()
        else:
            global_kv_cache_size = first_layer.length
        
        # # 모든 레이어의 토큰 수 확인
        # layer_tokens = []
        # for layer_kv in self.kv_cache:
        #     if hasattr(layer_kv, 'size'):
        #         layer_tokens.append(layer_kv.size())
        #     elif hasattr(layer_kv, 'length'):
        #         layer_tokens.append(layer_kv.length)
        
        # total_tokens = sum(layer_tokens) / len(layer_tokens) if layer_tokens else 0
        
        # 첫 번째 레이어의 상세 정보
        local_kv_size = first_layer.local_k.size(-2) if hasattr(first_layer, 'local_k') else 0
        global_remainder_size = first_layer.global_remainder[0].size(-2) if hasattr(first_layer, 'global_remainder') else 0
        num_global_blocks = first_layer.num_global_block if hasattr(first_layer, 'num_global_block') else 0
        init_exc = first_layer.init_exc if hasattr(first_layer, 'init_exc') else False
        
        # CPU 메모리 사용량
        cpu_memory_gb = self.calc_memory_usage() / (1024**3) if hasattr(self, 'calc_memory_usage') else 0.0
        
        return {
            'global_kv_cache_size': int(global_kv_cache_size),
            'local_kv_size': local_kv_size,
            'global_remainder_size': global_remainder_size,
            'num_global_blocks': num_global_blocks,
            'init_exc': init_exc,
            'cpu_memory_gb': cpu_memory_gb
        }
    
    def print_kv_cache_info(self):
        """KV 캐시 상태를 출력합니다."""
        info = self.get_kv_cache_info()
        logger.info("=" * 60)
        logger.info("KV Cache Status:")
        logger.info(f"  Accumulated global KV cache size: {info['global_kv_cache_size']}")
        logger.info(f"  Local KV cache size: {info['local_kv_size']} tokens")
        logger.info(f"  Global remainder size: {info['global_remainder_size']} tokens")
        logger.info(f"  Global blocks (on CPU): {info['num_global_blocks']}")
        logger.info(f"  Init KV cache full: {info['init_exc']}")
        logger.info(f"  CPU memory usage: {info['cpu_memory_gb']:.3f} GB")
        logger.info("=" * 60)

    @torch.inference_mode()
    def _get_video_features(self, pixel_values_videos):
        batch_size, frames, channels, height, width = pixel_values_videos.shape  # (B, Nv, 3, H, W)
        # Reshape to process frames individually
        pixel_values_videos = pixel_values_videos.view(batch_size * frames, channels, height, width)
        video_features = self.video_tower(pixel_values_videos, output_hidden_states=True)
        selected_video_feature = video_features.hidden_states[self.config.vision_feature_layer]
        
        if self.config.vision_feature_select_strategy == "default":
            selected_video_feature = selected_video_feature[:, 1:]
        elif self.config.vision_feature_select_strategy == "full":
            selected_video_feature = selected_video_feature
        
        video_features = self.multi_modal_projector(selected_video_feature)  # (Nv, 257, D)
        video_features = video_features.reshape(batch_size, frames * video_features.shape[1], -1)  # (B, Nv*257, D)
        return video_features
    
    @torch.inference_mode()
    def encode_video_chunk(self, video_chunk):
        """비디오 청크를 인코딩하여 video features를 추출합니다.
        
        Args:
            video_chunk: 비디오 청크 (Nv, H, W, 3)
            
        Returns:
            video_features: 인코딩된 비디오 features (1, Nv*256, D)
        """
        pixel_values_videos = self.processor.video_processor(images=None, videos=video_chunk, return_tensors="pt").pixel_values_videos.to(self.device, self.dtype)  # (1, Nv, 3, H, W)
        video_features = self._get_video_features(pixel_values_videos)  # (1, Nv*256, D)
        if self.n_local < video_features.shape[1]:
            logger.warning(f'n_local ({self.n_local}) is smaller than video_features tokens ({video_features.shape[1]}). Video will be truncated during prefill.')
        logger.debug(f'video_features: {video_features.shape[1]}')
        return video_features
    
    @torch.inference_mode()
    def video_prefill_chunk(self, video_features):
        """Video features를 language_model에 넣어서 KV cache를 업데이트합니다.
        
        Args:
            video_features: 인코딩된 비디오 features (1, Nv*256, D)
        """
        num_tokens = video_features.shape[1]
        
        # n_local보다 큰 경우 n_local 크기만큼 반복해서 모든 토큰 처리
        if num_tokens > self.n_local:
            logger.debug(f'video_features has {num_tokens} tokens, processing in chunks of {self.n_local}')
            start_idx = 0
            
            while start_idx < num_tokens:
                end_idx = min(start_idx + self.n_local, num_tokens)
                chunk_features = video_features[:, start_idx:end_idx, :]
                
                output = self.language_model(inputs_embeds=chunk_features, past_key_values=self.kv_cache, use_cache=True, return_dict=True)
                self.kv_cache = output.past_key_values
                
                logger.debug(f'Processed tokens {start_idx} to {end_idx} ({end_idx - start_idx} tokens)')
                start_idx = end_idx
        else:
            # n_local 이하인 경우 한 번에 처리
            output = self.language_model(inputs_embeds=video_features, past_key_values=self.kv_cache, use_cache=True, return_dict=True)
            self.kv_cache = output.past_key_values
        
        self.print_kv_cache_info()
        return
    
    @torch.inference_mode()
    def video_prefill(self, video_features_list):
        """Video features 리스트를 language_model에 넣어서 KV cache를 업데이트합니다.
        
        Args:
            video_features_list: 각 청크의 video features 리스트
        """
        super().video_prefill(video_features_list)
    
    @torch.inference_mode()
    def encode_and_prefill_video(self, video, encode_chunk_size=8, use_pipeline=False):  # video: (Nv, H, W, 3)
        """기존 호환성을 위한 래퍼 함수. encode_video와 video_prefill을 순차 호출합니다."""
        super().encode_and_prefill_video(video, encode_chunk_size, use_pipeline)

    @torch.inference_mode()
    def question_answering(self, input_text, max_new_tokens=128, retrieved_indices=None):
        device = self.device
        stop_token_ids = [self.processor.tokenizer.eos_token_id]

        output_ids = []
        stopped = False

        # NOTE: Only input the question to perform retrieval.
        input_ids = self.processor.tokenizer(input_text['question']).input_ids[1:]  # remove <s>
        input_ids = torch.as_tensor([input_ids], device=device)
        for layer_kv in self.kv_cache:  # retrieval mode
            layer_kv.set_retrieval()
        
        if retrieved_indices is None:  # Internal retrieval
            out = self.language_model(input_ids=input_ids, use_cache=True, past_key_values=self.kv_cache)
            past_key_values = out.past_key_values  # Retrieved KV-Cache: L x 2 x (B, h, N, Dh)
        else:  # External retrieval
            for layer_kv in self.kv_cache:
                assert layer_kv.block_size == self.n_frame_tokens, f'block_size: {layer_kv.block_size}, n_frame_tokens: {self.n_frame_tokens}'
                layer_kv.set_retrieved_block_indices(retrieved_indices)
            out = self.language_model(input_ids=input_ids, use_cache=True, past_key_values=self.kv_cache)
            past_key_values = out.past_key_values  # Retrieved KV-Cache: L x 2 x (B, h, N, Dh)
        
        for layer_kv in self.kv_cache:  # reset to default
            layer_kv.reset_retrieval()

        for i in range(max_new_tokens):
            if i == 0:  # prefill
                input_ids = self.processor.tokenizer(input_text['prompt']).input_ids[1:]  # remove <s>
                input_ids = torch.as_tensor([input_ids], device=device)
                inputs_embeds = self.get_input_embeddings()(input_ids)
                out = self.language_model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=past_key_values)
                past_key_values = out.past_key_values
                logits = out.logits
            else:  # decoding
                out = self.language_model(
                    input_ids=torch.as_tensor(
                        [[token]],
                        device=device,
                    ),
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                logits = out.logits
                past_key_values = out.past_key_values

            last_token_logits = logits[0, -1, :]
            
            _, indices = torch.topk(last_token_logits, 2)
            tokens = [int(index) for index in indices.tolist()]
            token = tokens[0]

            output_ids.append(token)

            if token in stop_token_ids:
                stopped = True
            else:
                stopped = False

            if i == max_new_tokens - 1 or stopped:
                break

        output = self.processor.tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            spaces_between_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )
        
        return output


def load_model(model_path='/mnt/models/Video-LLaVA-7B-hf', n_init=None, n_local=None, topk=8, chunk_size=1):
    device = 'cuda'
    n_frame_tokens = 256
    processor = VideoLlavaProcessor.from_pretrained(model_path)
    
    init_prompt = 'USER: '
    init_prompt_ids = processor.tokenizer(init_prompt, return_tensors="pt").input_ids.to(device)
    inf_llm_config = {
        'n_init': init_prompt_ids.shape[1] if n_init is None else n_init,
        'n_local': n_local,
        'fattn': True,
        'block_size': n_frame_tokens,
        'topk': topk,
        'chunk_size': chunk_size,
        'max_cached_block': 16,
        'exc_block_size': n_frame_tokens,
        'pin_memory': True,
    }
    model = VideoLlava_ReKV.from_pretrained(
        model_path, 
        device_map="auto",
        low_cpu_mem_usage=True, 
        torch_dtype=torch.float16,
        processor=processor,
        n_frame_tokens=n_frame_tokens,
        init_prompt_ids=init_prompt_ids,
        n_local=n_local,
        topk=topk,
        chunk_size=chunk_size,
    )
    
    model.language_model = patch_hf(model.language_model, **inf_llm_config)
    for k, v in inf_llm_config.items():
        logger.info(f'{k}: {v}')
    logger.info(f'n_frame_tokens: {n_frame_tokens}')

    model.eval()

    return model, processor

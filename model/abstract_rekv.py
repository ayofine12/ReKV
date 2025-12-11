import torch
from logzero import logger


class Abstract_ReKV:
    processor = None
    kv_cache = None

    def __init__(self, processor, n_frame_tokens, init_prompt_ids, n_local, topk, chunk_size):
        self.processor = processor
        self.n_frame_tokens = n_frame_tokens
        self.init_prompt_ids = init_prompt_ids
        self.n_local = n_local
        self.topk = topk
        self.chunk_size = chunk_size

    def clear_cache(self):
        self.kv_cache = None
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    @torch.inference_mode()
    def encode_init_prompt(self):
        if not isinstance(self.init_prompt_ids, torch.Tensor):
            self.init_prompt_ids = torch.as_tensor([self.init_prompt_ids], device=self.device)
        output = self.language_model(input_ids=self.init_prompt_ids, use_cache=True, return_dict=True)
        self.kv_cache = output.past_key_values
        return

    def _get_video_features(self, pixel_values_videos):
        pass

    def encode_video_chunk(self, video_chunk):
        """비디오 청크를 인코딩하여 video features를 추출합니다.
        
        Args:
            video_chunk: 비디오 청크 (Nv, H, W, 3)
            
        Returns:
            video_features: 인코딩된 비디오 features (1, Nv*196, D)
        """
        pixel_values_videos = self.processor.video_processor(video_chunk, return_tensors="pt").pixel_values_videos.to(self.device, self.dtype)  # (1, Nv, 3, H, W)
        video_features = self._get_video_features(pixel_values_videos)  # (1, Nv*196, D)
        assert self.n_local >= video_features.shape[1], f'n_local: {self.n_local}, video_features: {video_features.shape[1]}'
        return video_features
    
    def video_prefill_chunk(self, video_features):
        """Video features를 language_model에 넣어서 KV cache를 업데이트합니다.
        
        Args:
            video_features: 인코딩된 비디오 features (1, Nv*196, D)
        """
        output = self.language_model(inputs_embeds=video_features, past_key_values=self.kv_cache, use_cache=True, return_dict=True)
        self.kv_cache = output.past_key_values
        return
    
    def _encode_video_chunk(self, video_chunk):
        """기존 호환성을 위한 래퍼 함수. encode_video_chunk와 video_prefill_chunk를 순차 호출합니다."""
        video_features = self.encode_video_chunk(video_chunk)
        self.video_prefill_chunk(video_features)
        return

    @torch.inference_mode()
    def encode_video(self, video, encode_chunk_size=64):  # video: (Nv, H, W, 3)
        """비디오를 청크 단위로 인코딩하여 video features 리스트를 반환합니다.
        
        Args:
            video: 비디오 프레임들 (Nv, H, W, 3)
            encode_chunk_size: 청크 크기
            
        Returns:
            video_features_list: 각 청크의 video features 리스트
        """
        video_features_list = []
        num_frames = video.shape[0]
        num_chunks = num_frames // encode_chunk_size

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * encode_chunk_size
            end_idx = start_idx + encode_chunk_size
            chunk_video = video[start_idx:end_idx]
            video_features = self.encode_video_chunk(chunk_video)
            video_features_list.append(video_features)

        # Handle remaining frames
        remaining_frames = num_frames % encode_chunk_size
        if remaining_frames > 0:
            start_idx = num_chunks * encode_chunk_size
            end_idx = start_idx + remaining_frames
            remaining_video = video[start_idx:end_idx]
            video_features = self.encode_video_chunk(remaining_video)
            video_features_list.append(video_features)
        
        return video_features_list
    
    @torch.inference_mode()
    def video_prefill(self, video_features_list):
        """Video features 리스트를 language_model에 넣어서 KV cache를 업데이트합니다.
        
        Args:
            video_features_list: 각 청크의 video features 리스트
        """
        for video_features in video_features_list:
            self.video_prefill_chunk(video_features)
            logger.debug(f'KV-Cache RAM usage: {self.calc_memory_usage() / (1024**3):.3f} GB')
        
        logger.debug(f'KV-Cache RAM usage: {self.calc_memory_usage() / (1024**3):.1f} GB')
    
    @torch.inference_mode()
    def encode_and_prefill_video(self, video, encode_chunk_size=64):  # video: (Nv, H, W, 3)
        """기존 호환성을 위한 래퍼 함수. encode_video와 video_prefill을 순차 호출합니다.
        
        기존 코드와의 호환성을 위해 이 함수를 사용하거나, encode_video와 video_prefill을 분리하여 사용할 수 있습니다.
        """
        video_features_list = self.encode_video(video, encode_chunk_size)
        self.video_prefill(video_features_list)

    @torch.inference_mode()
    def question_answering(self, input_text, max_new_tokens=128):
        pass

    def calc_memory_usage(self):
        n_layers = len(self.kv_cache)
        memory = n_layers * self.kv_cache[0].calculate_cpu_memory()
        return memory

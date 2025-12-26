import torch
import copy
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration
from logzero import logger

from model.patch import patch_hf
from model.abstract_rekv import Abstract_ReKV

# NVTX for nsys profiling
try:
    import torch.cuda.nvtx as nvtx
    NVTX_AVAILABLE = True
    # NVTX 카테고리를 사용하여 색상 지정
    try:
        import ctypes
        libnvtx = ctypes.CDLL("libnvToolsExt.so.1")
        
        # 색상 정의 (ARGB 형식: 0xAARRGGBB)
        # 빨간색: 0xFFFF0000, 파란색: 0xFF0000FF
        NVTX_COLOR_RED = 0xFFFF0000
        NVTX_COLOR_BLUE = 0xFF0000FF
        
        # 카테고리 ID (0 = 기본 카테고리, 색상만 지정)
        NVTX_CATEGORY_RED = 0
        NVTX_CATEGORY_BLUE = 0
            
        # nvtxEventAttributes_t 구조체 정의 (NVTX v2/v3 호환)
        # payload는 union이지만 여기서는 uint64_t로 처리
        class NVTXEventAttributes(ctypes.Structure):
            _fields_ = [
                ("version", ctypes.c_uint16),      # 구조체 버전 (2 또는 3)
                ("size", ctypes.c_uint16),         # 구조체 크기
                ("category", ctypes.c_uint32),     # 카테고리 ID
                ("colorType", ctypes.c_int32),     # 색상 타입 (1 = ARGB)
                ("color", ctypes.c_uint32),        # 색상 값 (ARGB)
                ("payloadType", ctypes.c_int32),   # payload 타입
                ("payload", ctypes.c_uint64),       # payload 값 (union)
                ("messageType", ctypes.c_int32),   # 메시지 타입 (1 = ASCII)
                ("message", ctypes.c_char_p),      # 메시지 문자열
            ]
        
        # nvtxRangePushEx 함수 시그니처 수정
        libnvtx.nvtxRangePushEx.argtypes = [ctypes.POINTER(NVTXEventAttributes)]
        libnvtx.nvtxRangePushEx.restype = ctypes.c_int
        
        # 색상이 있는 range_push 함수
        def nvtx_range_push_colored(message, category_id, color):
            """색상이 지정된 NVTX 범위를 시작합니다."""
            attrs = NVTXEventAttributes()
            attrs.version = 3  # NVTX v3 구조체 사용
            attrs.size = ctypes.sizeof(NVTXEventAttributes)
            attrs.category = category_id
            attrs.colorType = 1  # NVTX_COLOR_ARGB = 1
            attrs.color = color
            attrs.payloadType = 0  # NVTX_PAYLOAD_UNKNOWN = 0
            attrs.payload = 0
            attrs.messageType = 1  # NVTX_MESSAGE_TYPE_ASCII = 1
            # 메시지는 null-terminated 문자열이어야 함
            msg_bytes = message.encode('utf-8') if isinstance(message, str) else message
            attrs.message = msg_bytes
            
            result = libnvtx.nvtxRangePushEx(ctypes.byref(attrs))
            if result != 0:  # NVTX_SUCCESS = 0
                # 실패 시 기본 range_push 사용
                nvtx.range_push(message)
        
        # 편의 함수들
        def nvtx_range_push_red(message):
            nvtx_range_push_colored(message, NVTX_CATEGORY_RED, NVTX_COLOR_RED)
        
        def nvtx_range_push_blue(message):
            nvtx_range_push_colored(message, NVTX_CATEGORY_BLUE, NVTX_COLOR_BLUE)
        
        # nvtx 모듈에 함수 추가
        nvtx.range_push_red = nvtx_range_push_red
        nvtx.range_push_blue = nvtx_range_push_blue
        
    except Exception as e:
        # ctypes 사용 실패 시 기본 nvtx만 사용
        logger.warning(f"Failed to setup colored NVTX ranges: {e}. Using default NVTX.")
        # 기본 함수들 (색상 없음)
        nvtx.range_push_red = nvtx.range_push
        nvtx.range_push_blue = nvtx.range_push
        
except ImportError:
    NVTX_AVAILABLE = False
    # Create dummy context manager if NVTX is not available
    class nvtx:
        @staticmethod
        def range_push(msg):
            pass
        @staticmethod
        def range_pop():
            pass
        @staticmethod
        def range_push_red(msg):
            pass
        @staticmethod
        def range_push_blue(msg):
            pass


class VideoLlava_ReKV(VideoLlavaForConditionalGeneration, Abstract_ReKV):
    def __init__(self, config, processor, n_frame_tokens, init_prompt_ids, n_local, topk, chunk_size):
        VideoLlavaForConditionalGeneration.__init__(self, config)
        Abstract_ReKV.__init__(self, processor, n_frame_tokens, init_prompt_ids, n_local, topk, chunk_size)
        self.processor.video_processor = self.processor.image_processor
        
        # language_model은 GPU 0에만 배치되어야 함 (encoding 제외한 모든 작업)
        # 초기화 후 명시적으로 GPU 0으로 이동
        self._ensure_language_model_on_gpu0 = True
        
        # GPU 분산 처리를 위한 설정
        self.use_multi_gpu_encoding = False
        # self.enable_multi_gpu_encoding = True  # GPU 분산 인코딩 활성화 여부 (수동 설정 가능)
        self.video_tower_gpu0 = None
        self.video_tower_gpu1 = None
        self.multi_modal_projector_gpu0 = None
        self.multi_modal_projector_gpu1 = None
        
        if torch.cuda.device_count() >= 2:
            self.device_0 = torch.device("cuda:0")
            self.device_1 = torch.device("cuda:1")
            self.use_multi_gpu_encoding = True
            logger.info(f"Multi-GPU encoding available: GPU 0 and GPU 1 available")
    
    def clear_cache(self):
        """KV cache를 정리하고 GPU 메모리를 비웁니다."""
        self.kv_cache = None
        
        # GPU 0과 1의 캐시 모두 정리
        torch.cuda.empty_cache()
        if self.use_multi_gpu_encoding:
            with torch.cuda.device(self.device_0):
                torch.cuda.empty_cache()
            with torch.cuda.device(self.device_1):
                torch.cuda.empty_cache()
        
        torch.cuda.ipc_collect()

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
    def encode_video_chunk(self, video_chunk, enable_multi_gpu_encoding):
        """비디오 청크를 인코딩하여 video features를 추출합니다.
        
        Args:
            video_chunk: 비디오 청크 (Nv, H, W, 3)
            
        Returns:
            video_features: 인코딩된 비디오 features (1, Nv*256, D)
        """
        if NVTX_AVAILABLE:
            # 빨간색으로 표시
            nvtx.range_push_red("encode_video_chunk")
        try:
            num_frames = video_chunk.shape[0]
            
            # GPU 분산 인코딩 설정
            # chunk_size > 16이고 GPU가 2개 이상일 때 분산 처리
            # 주의: GPU 분산 인코딩은 video_tower를 GPU 1로 이동시킬 때 내부 버퍼/상태 동기화 문제로 인해
            # vectorized_gather_kernel 인덱스 오류가 발생할 수 있습니다.
            # self.enable_multi_gpu_encoding을 True로 설정하여 활성화할 수 있습니다.
            
            if enable_multi_gpu_encoding and self.chunk_size > 16 and self.use_multi_gpu_encoding and num_frames > 1:
                result = self._encode_video_chunk_multi_gpu(video_chunk)
            else:
                # 단일 GPU 처리 (GPU 0에서만 실행)
                pixel_values_videos = self.processor.video_processor(images=None, videos=video_chunk, return_tensors="pt").pixel_values_videos.to(self.device, self.dtype)  # (1, Nv, 3, H, W)
                result = self._get_video_features(pixel_values_videos)  # (1, Nv*256, D)
            return result
        finally:
            if NVTX_AVAILABLE:
                nvtx.range_pop()
    
    @torch.inference_mode()
    def _encode_video_chunk_multi_gpu(self, video_chunk):
        """GPU 0과 1에 프레임을 분배하여 인코딩합니다.
        
        Args:
            video_chunk: 비디오 청크 (Nv, H, W, 3)
            
        Returns:
            video_features: 인코딩된 비디오 features (1, Nv*256, D) - GPU 0에 위치
        """
        num_frames = video_chunk.shape[0]
        
        # 프레임을 두 부분으로 분할
        mid_point = num_frames // 2
        chunk_gpu0 = video_chunk[:mid_point]
        chunk_gpu1 = video_chunk[mid_point:]
        
        # GPU 0에서 처리
        with torch.cuda.device(self.device_0):
            pixel_values_gpu0 = self.processor.video_processor(images=None, videos=chunk_gpu0, return_tensors="pt").pixel_values_videos.to(self.device_0, self.dtype)
            features_gpu0 = self._get_video_features_on_device(pixel_values_gpu0, self.device_0)
            torch.cuda.synchronize(self.device_0)
        
        # GPU 1에서 처리
        with torch.cuda.device(self.device_1):
            pixel_values_gpu1 = self.processor.video_processor(images=None, videos=chunk_gpu1, return_tensors="pt").pixel_values_videos.to(self.device_1, self.dtype)
            features_gpu1 = self._get_video_features_on_device(pixel_values_gpu1, self.device_1)
            torch.cuda.synchronize(self.device_1)
        
        # GPU 1의 결과를 GPU 0으로 이동하고 연결
        with torch.cuda.device(self.device_0):
            features_gpu1 = features_gpu1.to(self.device_0)
            video_features = torch.cat([features_gpu0, features_gpu1], dim=1)  # (1, Nv*256, D)
            torch.cuda.synchronize(self.device_0)
        
        logger.debug(f'Multi-GPU encoding: GPU0 processed {mid_point} frames, GPU1 processed {num_frames - mid_point} frames')
        
        return video_features
    
    def _ensure_models_on_devices(self):
        """각 GPU에 모델을 배치합니다 (lazy initialization).
        
        GPU 1에는 모델의 완전한 복사본을 생성하여 배치합니다.
        이렇게 하면 매번 모델을 이동시킬 필요가 없어 효율적입니다.
        """
        if not self.use_multi_gpu_encoding:
            return
        
        if self.video_tower_gpu0 is None:
            # GPU 0에 모델 배치 (원본 모델 사용)
            self.video_tower_gpu0 = self.video_tower
            self.multi_modal_projector_gpu0 = self.multi_modal_projector
            logger.debug("Models reference set for GPU 0")
        
        if self.video_tower_gpu1 is None:
            # GPU 1에 모델의 완전한 복사본 생성 및 배치
            with torch.cuda.device(self.device_1):
                # 모델을 GPU 1로 복사 (deep copy)
                self.video_tower_gpu1 = copy.deepcopy(self.video_tower).to(self.device_1)
                self.multi_modal_projector_gpu1 = copy.deepcopy(self.multi_modal_projector).to(self.device_1)
                torch.cuda.synchronize(self.device_1)
            logger.info("Models copied and placed on GPU 1 (permanent placement)")
    
    @torch.inference_mode()
    def _get_video_features_on_device(self, pixel_values_videos, device):
        """특정 디바이스에서 비디오 features를 추출합니다.
        
        Args:
            pixel_values_videos: 픽셀 값 (B, Nv, 3, H, W)
            device: 대상 디바이스
            
        Returns:
            video_features: 인코딩된 비디오 features (B, Nv*257, D)
        """
        # 모델이 해당 디바이스에 배치되어 있는지 확인
        self._ensure_models_on_devices()
        
        batch_size, frames, channels, height, width = pixel_values_videos.shape
        pixel_values_videos = pixel_values_videos.view(batch_size * frames, channels, height, width)
        
        # 해당 디바이스의 모델 사용
        if device == self.device_0:
            video_tower_device = self.video_tower_gpu0
            multi_modal_projector_device = self.multi_modal_projector_gpu0
        elif device == self.device_1:
            video_tower_device = self.video_tower_gpu1
            multi_modal_projector_device = self.multi_modal_projector_gpu1
        else:
            # Fallback: 모델을 해당 디바이스로 이동
            video_tower_device = self.video_tower.to(device)
            multi_modal_projector_device = self.multi_modal_projector.to(device)
        
        # 해당 디바이스의 CUDA 컨텍스트에서 실행
        # GPU 1에는 이미 모델이 배치되어 있으므로 바로 사용
        with torch.cuda.device(device):
            # 모델이 해당 디바이스에 있는지 확인 (GPU 1의 경우 이미 배치되어 있어야 함)
            actual_device = next(video_tower_device.parameters()).device
            if actual_device != device:
                logger.warning(f"video_tower_device is on {actual_device}, moving to {device}")
                video_tower_device = video_tower_device.to(device)
                multi_modal_projector_device = multi_modal_projector_device.to(device)
            
            video_features = video_tower_device(pixel_values_videos, output_hidden_states=True)
            
            # hidden_states가 제대로 생성되었는지 확인
            if not hasattr(video_features, 'hidden_states') or video_features.hidden_states is None:
                raise RuntimeError(f"video_tower on {device} did not return hidden_states")
            
            # vision_feature_layer 인덱스가 유효한지 확인
            num_layers = len(video_features.hidden_states)
            if self.config.vision_feature_layer >= num_layers:
                logger.warning(f"vision_feature_layer {self.config.vision_feature_layer} >= num_layers {num_layers}, using last layer")
                layer_idx = num_layers - 1
            else:
                layer_idx = self.config.vision_feature_layer
            
            selected_video_feature = video_features.hidden_states[layer_idx]
            
            if self.config.vision_feature_select_strategy == "default":
                selected_video_feature = selected_video_feature[:, 1:]
            elif self.config.vision_feature_select_strategy == "full":
                selected_video_feature = selected_video_feature
            
            video_features = multi_modal_projector_device(selected_video_feature)
            
            # GPU 동기화
            torch.cuda.synchronize(device)
        
        video_features = video_features.reshape(batch_size, frames * video_features.shape[1], -1)
        return video_features
    
    @torch.inference_mode()
    def video_prefill_chunk(self, video_features):
        """Video features를 language_model에 넣어서 KV cache를 업데이트합니다.
        
        Args:
            video_features: 인코딩된 비디오 features (1, Nv*256, D)
        """
        if NVTX_AVAILABLE:
            # 파란색으로 표시
            nvtx.range_push_blue("video_prefill_chunk")
        try:
            num_tokens = video_features.shape[1]
            
            # n_local보다 큰 경우 n_local 크기만큼 반복해서 모든 토큰 처리
            if num_tokens > self.n_local:
                start_idx = 0
                
                while start_idx < num_tokens:
                    end_idx = min(start_idx + self.n_local, num_tokens)
                    chunk_features = video_features[:, start_idx:end_idx, :]
                    
                    output = self.language_model(inputs_embeds=chunk_features, past_key_values=self.kv_cache, use_cache=True, return_dict=True)
                    self.kv_cache = output.past_key_values
                    
                    start_idx = end_idx
            else:
                # n_local 이하인 경우 한 번에 처리
                output = self.language_model(inputs_embeds=video_features, past_key_values=self.kv_cache, use_cache=True, return_dict=True)
                self.kv_cache = output.past_key_values
            
            self.print_kv_cache_info()
        finally:
            if NVTX_AVAILABLE:
                nvtx.range_pop()
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
    device = 'cuda:0'  # GPU 0으로 명시적으로 설정
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
    # language_model은 GPU 0에만 배치
    model = VideoLlava_ReKV.from_pretrained(
        model_path, 
        device_map={"": device},  # 모든 모델을 GPU 0에 배치
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
    
    # language_model을 GPU 0에 명시적으로 배치 (encoding 제외한 모든 작업은 GPU 0에서 수행)
    model.language_model = model.language_model.to(device)
    logger.info(f"language_model placed on {device}")
    
    # language_model의 모든 파라미터가 GPU 0에 있는지 확인
    for name, param in model.language_model.named_parameters():
        if param.device.type == 'cuda' and param.device.index != 0:
            logger.warning(f"Moving {name} from {param.device} to {device}")
            param.data = param.data.to(device)
    for k, v in inf_llm_config.items():
        logger.info(f'{k}: {v}')
    logger.info(f'n_frame_tokens: {n_frame_tokens}')

    model.eval()

    return model, processor

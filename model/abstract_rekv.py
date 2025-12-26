import torch
import threading
import queue
import time
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
        self.encode_times = []
        self.prefill_times = []
        self._times_lock = threading.Lock()

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
        if self.n_local < video_features.shape[1]:
            logger.warning(f'n_local ({self.n_local}) is smaller than video_features tokens ({video_features.shape[1]}). Video will be truncated during prefill.')
        return video_features
    
    def video_prefill_chunk(self, video_features):
        """Video features를 language_model에 넣어서 KV cache를 업데이트합니다.
        
        Args:
            video_features: 인코딩된 비디오 features (1, Nv*196, D)
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
        
        return
    
    def _encode_and_prefill_video_chunk(self, video_chunk):
        """기존 호환성을 위한 래퍼 함수. encode_video_chunk와 video_prefill_chunk를 순차 호출합니다."""
        video_features = self.encode_video_chunk(video_chunk)
        self.video_prefill_chunk(video_features)
        return

    @torch.inference_mode()
    def encode_and_prefill_video(self, video, encode_chunk_size=64, use_pipeline=False):  # video: (Nv, H, W, 3)
        """비디오를 청크 단위로 인코딩하여 video features 리스트를 반환합니다.
        
        Args:
            video: 비디오 프레임들 (Nv, H, W, 3)
            encode_chunk_size: 청크 크기
            use_pipeline: 파이프라인 모드 사용 여부 (encoding과 prefill을 병렬 처리)
            
        Returns:
            video_features_list: 각 청크의 video features 리스트
        """
        # 시간 수집 시작
        self._collecting_times = True
        self.encode_times = []
        self.prefill_times = []
        
        try:
            if use_pipeline:
                result = self._encode_and_prefill_video_pipeline(video, encode_chunk_size)
            else:
                result = self._encode_and_prefill_video_sequential(video, encode_chunk_size)
            
            # 시간 통계 출력
            self._print_timing_stats()
            
            return result
        finally:
            # 시간 수집 종료
            self._collecting_times = False
    
    def _print_timing_stats(self):
        """시간 통계를 출력합니다."""
        if not self.encode_times and not self.prefill_times:
            return
        
        print("\n" + "="*60)
        print("Video Encoding and Prefill Timing Statistics")
        print("="*60)
        
        if self.encode_times:
            print(f"\n[encode_video_chunk] Times (seconds):")
            for i, t in enumerate(self.encode_times):
                print(f"  Chunk {i}: {t:.4f}")
            print(f"  Total: {sum(self.encode_times):.4f}")
            print(f"  Average: {sum(self.encode_times)/len(self.encode_times):.4f}")
            print(f"  Min: {min(self.encode_times):.4f}")
            print(f"  Max: {max(self.encode_times):.4f}")
        
        if self.prefill_times:
            print(f"\n[video_prefill_chunk] Times (seconds):")
            for i, t in enumerate(self.prefill_times):
                print(f"  Chunk {i}: {t:.4f}")
            print(f"  Total: {sum(self.prefill_times):.4f}")
            print(f"  Average: {sum(self.prefill_times)/len(self.prefill_times):.4f}")
            print(f"  Min: {min(self.prefill_times):.4f}")
            print(f"  Max: {max(self.prefill_times):.4f}")
        
        if self.encode_times and self.prefill_times:
            total_time = sum(self.encode_times) + sum(self.prefill_times)
            print(f"\n[Total] Combined time: {total_time:.4f} seconds")
        
        print("="*60 + "\n")
    
    def _encode_and_prefill_video_sequential(self, video, encode_chunk_size):
        """Sequential 방식: encoding과 prefill을 순차적으로 처리합니다."""
        video_features_list = []
        num_frames = video.shape[0]
        num_chunks = num_frames // encode_chunk_size

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * encode_chunk_size
            end_idx = start_idx + encode_chunk_size
            chunk_video = video[start_idx:end_idx]
            video_features = self._encode_and_prefill_video_chunk(chunk_video)
            video_features_list.append(video_features)

        # Handle remaining frames
        remaining_frames = num_frames % encode_chunk_size
        if remaining_frames > 0:
            start_idx = num_chunks * encode_chunk_size
            end_idx = start_idx + remaining_frames
            remaining_video = video[start_idx:end_idx]
            video_features = self._encode_and_prefill_video_chunk(remaining_video)
            video_features_list.append(video_features)
        
        return video_features_list
    
    def _encode_and_prefill_video_pipeline(self, video, encode_chunk_size):
        """Pipeline 방식: encoding과 prefill을 병렬로 처리합니다.
        
        - chunk n이 prefill되고 있을 때 chunk n+1이 encoding 됨
        - queue의 최대 크기는 2
        - semaphore를 사용해서 queue가 가득 차면 encoding이 대기
        """
        num_frames = video.shape[0]
        num_chunks = num_frames // encode_chunk_size
        has_remaining = (num_frames % encode_chunk_size) > 0
        total_chunks = num_chunks + (1 if has_remaining else 0)
        
        # Queue for passing encoded features from encoding thread to prefill thread
        # Max size 2: allows encoding to be 2 chunks ahead of prefill
        feature_queue = queue.Queue(maxsize=2)
        
        # Semaphore to limit queue size (2 slots available)
        queue_semaphore = threading.Semaphore(2)
        
        # Event to signal completion
        encoding_done = threading.Event()
        prefill_done = threading.Event()
        exception_occurred = threading.Event()
        exception_info = [None]
        
        video_features_list = []
        
        def encoding_worker():
            """Encoding thread: encodes video chunks and puts them in the queue."""
            try:
                # Process regular chunks
                for chunk_idx in range(num_chunks):
                    # Wait for queue slot to be available (semaphore)
                    queue_semaphore.acquire()
                    
                    start_idx = chunk_idx * encode_chunk_size
                    end_idx = start_idx + encode_chunk_size
                    chunk_video = video[start_idx:end_idx]
                    
                    # Encode chunk
                    video_features = self.encode_video_chunk(chunk_video)
                    
                    # Put in queue (this will block if queue is full, but semaphore prevents this)
                    feature_queue.put((chunk_idx, video_features))
                    logger.debug(f'Encoded chunk {chunk_idx}, queue size: {feature_queue.qsize()}')
                
                # Handle remaining frames
                if has_remaining:
                    queue_semaphore.acquire()
                    start_idx = num_chunks * encode_chunk_size
                    end_idx = start_idx + (num_frames % encode_chunk_size)
                    remaining_video = video[start_idx:end_idx]
                    video_features = self.encode_video_chunk(remaining_video)
                    feature_queue.put((num_chunks, video_features))
                    logger.debug(f'Encoded remaining chunk, queue size: {feature_queue.qsize()}')
                
                encoding_done.set()
                logger.debug('Encoding thread finished')
                
            except Exception as e:
                logger.error(f'Encoding thread error: {e}', exc_info=True)
                exception_info[0] = e
                exception_occurred.set()
                encoding_done.set()
        
        def prefill_worker():
            """Prefill thread: takes encoded features from queue and performs prefill."""
            try:
                processed_chunks = 0
                results = {}
                
                while processed_chunks < total_chunks:
                    # Get encoded features from queue (blocks until available)
                    # No timeout - wait indefinitely until item is available or encoding is done
                    chunk_idx, video_features = feature_queue.get()
                    
                    # Perform prefill
                    self.video_prefill_chunk(video_features)
                    
                    # Store result
                    results[chunk_idx] = video_features
                    processed_chunks += 1
                    
                    # Release semaphore slot
                    queue_semaphore.release()
                    
                    logger.debug(f'Prefilled chunk {chunk_idx}, processed: {processed_chunks}/{total_chunks}')
                
                # Sort results by chunk index
                for idx in sorted(results.keys()):
                    video_features_list.append(results[idx])
                
                prefill_done.set()
                logger.debug('Prefill thread finished')
                
            except Exception as e:
                logger.error(f'Prefill thread error: {e}', exc_info=True)
                exception_info[0] = e
                exception_occurred.set()
                prefill_done.set()
        
        # Start threads
        encoding_thread = threading.Thread(target=encoding_worker, daemon=True)
        prefill_thread = threading.Thread(target=prefill_worker, daemon=True)
        
        encoding_thread.start()
        prefill_thread.start()
        
        # Wait for both threads to complete
        encoding_thread.join()
        prefill_thread.join()
        
        # Check for exceptions
        if exception_occurred.is_set():
            raise RuntimeError(f"Exception in pipeline threads: {exception_info[0]}") from exception_info[0]
        
        return video_features_list

    @torch.inference_mode()
    def question_answering(self, input_text, max_new_tokens=128):
        pass

    def calc_memory_usage(self):
        n_layers = len(self.kv_cache)
        memory = n_layers * self.kv_cache[0].calculate_cpu_memory()
        return memory

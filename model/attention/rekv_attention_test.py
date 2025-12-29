import copy
import torch
from typing import Optional
from pathlib import Path
import threading

from .kv_cache_manager import ContextManager
from .position_bias_utils import load_position_bias

def copy_context_manager_to_device(src_manager: ContextManager, target_device: str):
    """
    Copy ContextManager state from source device to target device.
    This allows pre-transferring KV cache to another GPU for pipelined processing.
    """
    # Create position_bias on target device
    target_position_bias = copy.deepcopy(src_manager.position_embedding)
    if target_position_bias._cos_cached is not None:
        target_position_bias._cos_cached = target_position_bias._cos_cached.to(target_device)
    if target_position_bias._sin_cached is not None:
        target_position_bias._sin_cached = target_position_bias._sin_cached.to(target_device)
    if hasattr(target_position_bias, 'inv_freq'):
        target_position_bias.inv_freq = target_position_bias.inv_freq.to(target_device)
    
    # Create new ContextManager on target device
    target_manager = ContextManager(
        target_position_bias,
        src_manager.n_init, src_manager.n_local,
        src_manager.block_size, src_manager.max_cached_block, 
        src_manager.topk, src_manager.chunk_size, src_manager.exc_block_size,
        src_manager.fattn,
        src_manager.async_global_stream,
        src_manager.pin_memory,
    )
    
    # Copy state if initialized
    if src_manager.initialized:
        # Copy local KV cache
        target_manager.local_k = src_manager.local_k.to(target_device)
        target_manager.local_v = src_manager.local_v.to(target_device)
        
        # Copy global remainder
        target_manager.global_remainder = (
            src_manager.global_remainder[0].to(target_device),
            src_manager.global_remainder[1].to(target_device),
        )
        target_manager._global_remainder_st = src_manager._global_remainder_st
        target_manager._global_remainder_ed = src_manager._global_remainder_ed
        
        # Copy init KV
        target_manager.init_k = src_manager.init_k.to(target_device)
        target_manager.init_v = src_manager.init_v.to(target_device)
        target_manager.init_exc = src_manager.init_exc
        
        # Copy length
        target_manager.length = src_manager.length
        
        # Copy global blocks (this is more complex, but for now we'll skip it)
        # The global_blocks will be loaded from CPU when needed via retrieval
        
        # Mark as initialized
        target_manager.initialized = True
    
    return target_manager

def __main__():
    # Load position_bias for GPU 0
    position_bias_gpu0 = load_position_bias("/root/mwnoh/ReKV/model/attention/position_bias.pkl", device="cuda:0")
    # Load position_bias for GPU 1  
    position_bias_gpu1 = load_position_bias("/root/mwnoh/ReKV/model/attention/position_bias.pkl", device="cuda:1")
    
    n_init = 5
    n_local = 4096
    block_size = 256
    max_cached_block = 16
    topk = 64
    chunk_size = 32
    exc_block_size = 256
    fattn = True
    async_global_stream = True
    pin_memory = True
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    # Create ContextManager for GPU 0
    past_key_value_gpu0 = ContextManager(
        position_bias_gpu0,
        n_init, n_local, 
        block_size, max_cached_block, topk, chunk_size, exc_block_size,
        fattn,
        async_global_stream,
        pin_memory,
    )
    
    # Create ContextManager for GPU 1
    past_key_value_gpu1 = ContextManager(
        position_bias_gpu1,
        n_init, n_local, 
        block_size, max_cached_block, topk, chunk_size, exc_block_size,
        fattn,
        async_global_stream,
        pin_memory,
    )

    batch_size = 1
    len_q = 4096
    num_heads = 32
    dim_head = 128
    
    # First batch on GPU 0
    print("Processing first batch on GPU 0...")
    h_q_gpu0 = torch.randn(batch_size, num_heads, n_init+len_q, dim_head, dtype=torch.float16, device="cuda:0")
    h_k_gpu0 = torch.randn(batch_size, num_heads, n_init+len_q, dim_head, dtype=torch.float16, device="cuda:0")
    h_v_gpu0 = torch.randn(batch_size, num_heads, n_init+len_q, dim_head, dtype=torch.float16, device="cuda:0")

    local_q_gpu0, local_k_gpu0, local_v_gpu0 = h_q_gpu0, h_k_gpu0, h_v_gpu0
    global_q_gpu0, global_k_gpu0, global_v_gpu0 = h_q_gpu0, h_k_gpu0, h_v_gpu0

    h_q_gpu1 = torch.randn(batch_size, num_heads, len_q, dim_head, dtype=torch.float16, device="cuda:1")
    h_k_gpu1 = torch.randn(batch_size, num_heads, len_q, dim_head, dtype=torch.float16, device="cuda:1")
    h_v_gpu1 = torch.randn(batch_size, num_heads, len_q, dim_head, dtype=torch.float16, device="cuda:1")

    local_q_gpu1, local_k_gpu1, local_v_gpu1 = h_q_gpu1, h_k_gpu1, h_v_gpu1
    global_q_gpu1, global_k_gpu1, global_v_gpu1 = h_q_gpu1, h_k_gpu1, h_v_gpu1

    # # Initialize both GPU0 and GPU1 before timing
    # if not past_key_value_gpu0.initialized:
    #     past_key_value_gpu0.init(local_q_gpu0, local_k_gpu0, local_v_gpu0,
    #                              global_q_gpu0, global_k_gpu0, global_v_gpu0)
    
    o_gpu0 = past_key_value_gpu0.append(
        local_q_gpu0, local_k_gpu0, local_v_gpu0,
        global_q_gpu0, global_k_gpu0, global_v_gpu0,
    )

    if not past_key_value_gpu1.initialized:
        past_key_value_gpu1.init(local_q_gpu1, local_k_gpu1, local_v_gpu1, 
                                 global_q_gpu1, global_k_gpu1, global_v_gpu1)

    h_q_gpu0 = torch.randn(batch_size, num_heads, len_q, dim_head, dtype=torch.float16, device="cuda:0")
    h_k_gpu0 = torch.randn(batch_size, num_heads, len_q, dim_head, dtype=torch.float16, device="cuda:0")
    h_v_gpu0 = torch.randn(batch_size, num_heads, len_q, dim_head, dtype=torch.float16, device="cuda:0")

    local_q_gpu0, local_k_gpu0, local_v_gpu0 = h_q_gpu0, h_k_gpu0, h_v_gpu0
    global_q_gpu0, global_k_gpu0, global_v_gpu0 = h_q_gpu0, h_k_gpu0, h_v_gpu0


    # Create CUDA streams
    gpu0_stream = torch.cuda.Stream(device=0)
    gpu1_stream = torch.cuda.Stream(device=1)
    
    # Events for timing
    append_gpu0_start = torch.cuda.Event(enable_timing=True)
    append_gpu0_end = torch.cuda.Event(enable_timing=True)
    transfer_start = torch.cuda.Event(enable_timing=True)
    transfer_end = torch.cuda.Event(enable_timing=True)
    append_gpu1_start = torch.cuda.Event(enable_timing=True)
    append_gpu1_end = torch.cuda.Event(enable_timing=True)
    
    # Shared flags for synchronization
    transfer_ready = threading.Event()  # Signal to start transferring input tensors
    append_complete = threading.Event()  # Signal that GPU0 append is complete
    transfer_complete = threading.Event()  # Signal that all transfers are complete
    
    # Synchronize before starting
    torch.cuda.synchronize(device=0)
    torch.cuda.synchronize(device=1)
    
    # Define transfer function to run in parallel thread
    def transfer_to_gpu1():
        """Transfer KV cache from GPU 0 to GPU 1 in two stages:
        Stage 1: Transfer input tensors (local_k_gpu0, local_v_gpu0) immediately when append starts
        Stage 2: Transfer all updated states after append completes
        """
        # Stage 1: Transfer input tensors as soon as append starts
        transfer_ready.wait()
        
        print("Stage 1: Transferring input tensors (local_k_gpu0, local_v_gpu0) to GPU 1...")
        with torch.cuda.device(1):
            with torch.cuda.stream(gpu1_stream):
                transfer_start.record(stream=gpu1_stream)
                torch.cuda.nvtx.range_push("GPU0_to_GPU1_KV_Cache_Transfer")
                
                # Transfer input tensors local_k_gpu0 and local_v_gpu0 to GPU1
                # Concatenate with existing local_k and local_v
                local_k_gpu1_new = local_k_gpu0.to("cuda:1", non_blocking=True)
                local_v_gpu1_new = local_v_gpu0.to("cuda:1", non_blocking=True)
                
                past_key_value_gpu1.local_k = torch.cat((past_key_value_gpu1.local_k, local_k_gpu1_new), dim=-2)
                past_key_value_gpu1.local_v = torch.cat((past_key_value_gpu1.local_v, local_v_gpu1_new), dim=-2)
                
                # Apply n_local restriction (same as in ContextManager.append)
                if past_key_value_gpu1.local_k.size(-2) >= past_key_value_gpu1.n_local:
                    past_key_value_gpu1.local_k = past_key_value_gpu1.local_k[:, :, -past_key_value_gpu1.n_local:, :]
                    past_key_value_gpu1.local_v = past_key_value_gpu1.local_v[:, :, -past_key_value_gpu1.n_local:, :]
                
                torch.cuda.nvtx.range_pop()
        
        # Stage 2: Wait for GPU0 append to complete, then transfer all updated states
        append_complete.wait()
        
        print("Stage 2: Transferring updated states after GPU 0 append completion...")
        with torch.cuda.device(1):
            with torch.cuda.stream(gpu1_stream):
                torch.cuda.nvtx.range_push("GPU0_to_GPU1_State_Transfer")
                
                # Copy updated states from GPU0's append
                # (Metadata like batch_size, num_heads, dtype, etc. are already set by init())
                
                # Copy global remainder (updated during append)
                past_key_value_gpu1.global_remainder = (
                    past_key_value_gpu0.global_remainder[0].to("cuda:1", non_blocking=True),
                    past_key_value_gpu0.global_remainder[1].to("cuda:1", non_blocking=True),
                )
                past_key_value_gpu1._global_remainder_st = past_key_value_gpu0._global_remainder_st
                past_key_value_gpu1._global_remainder_ed = past_key_value_gpu0._global_remainder_ed
                
                # Copy init KV (may be updated during append)
                past_key_value_gpu1.init_k = past_key_value_gpu0.init_k.to("cuda:1", non_blocking=True)
                past_key_value_gpu1.init_v = past_key_value_gpu0.init_v.to("cuda:1", non_blocking=True)
                past_key_value_gpu1.init_exc = past_key_value_gpu0.init_exc
                
                # Copy length (updated during append)
                past_key_value_gpu1.length = past_key_value_gpu0.length
                
                # Copy num_global_block (increases when new blocks are created during append)
                past_key_value_gpu1.num_global_block = past_key_value_gpu0.num_global_block
                
                # Copy global_blocks (CPU-based, can be shared)
                # Note: This is a shallow copy - MemoryUnits reference GPU0's cuda_cache
                # When GPU1 does retrieval, it will load blocks from CPU to GPU1's cache
                past_key_value_gpu1.global_blocks = past_key_value_gpu0.global_blocks
                
                # Copy block_k data (representative keys) from GPU0 to GPU1
                for u in range(past_key_value_gpu0.num_units):
                    if past_key_value_gpu0.block_k[u].length > 0:
                        gpu0_data = past_key_value_gpu0.block_k[u].get_data()  # (length, hidden_size)
                        gpu1_data = gpu0_data.to("cuda:1", non_blocking=True)
                        past_key_value_gpu1.block_k[u].append(gpu1_data)  # Append all at once
                
                # Note: The following are already created by init() and don't need to be copied:
                # - batch_size, num_heads, num_heads_kv, dim_head, num_units, unit_size, unit_size_kv, dtype
                # - cuda_cache (independent per GPU)
                
                torch.cuda.nvtx.range_pop()
                transfer_end.record(stream=gpu1_stream)
        
        transfer_complete.set()
    
    # Start transfer thread
    transfer_thread = threading.Thread(target=transfer_to_gpu1)
    transfer_thread.start()
    
    # Process batch on GPU 0
    print("Processing batch on GPU 0...")
    print(f"\nBefore GPU0 append (second batch):")
    print(f"  GPU0 num_global_block: {past_key_value_gpu0.num_global_block}")
    print(f"  GPU0 global_blocks length: {len(past_key_value_gpu0.global_blocks[0])}")
    print(f"  GPU0 global_remainder shape: {past_key_value_gpu0.global_remainder[0].shape}")
    
    # Signal transfer thread to start Stage 1 (input tensor transfer) immediately
    print("Signaling transfer thread to begin Stage 1 (input tensors)...")
    transfer_ready.set()
    
    with torch.cuda.device(0):
        with torch.cuda.stream(gpu0_stream):
            append_gpu0_start.record(stream=gpu0_stream)
            torch.cuda.nvtx.range_push("GPU0_Append_Overlapped")
            
            o_gpu0 = past_key_value_gpu0.append(
                local_q_gpu0, local_k_gpu0, local_v_gpu0,
                global_q_gpu0, global_k_gpu0, global_v_gpu0,
            )
            
            torch.cuda.nvtx.range_pop()
            append_gpu0_end.record(stream=gpu0_stream)
    
    print(f"\nAfter GPU0 append (second batch):")
    print(f"  GPU0 num_global_block: {past_key_value_gpu0.num_global_block}")
    print(f"  GPU0 global_blocks length: {len(past_key_value_gpu0.global_blocks[0])}")
    
    # Signal that GPU0 append is complete - start Stage 2 transfer
    print("GPU 0 append complete - signaling Stage 2 transfer...")
    append_complete.set()
    
    # Wait for both GPU 0 append and transfer to complete
    print("Waiting for transfer to complete...")
    transfer_thread.join()
    torch.cuda.synchronize(device=0)
    torch.cuda.synchronize(device=1)
    
    # Calculate timing
    append_gpu0_time_ms = append_gpu0_start.elapsed_time(append_gpu0_end)
    transfer_time_ms = transfer_start.elapsed_time(transfer_end)
    
    print(f"\n{'='*60}")
    print(f"Overlapped Timing Results:")
    print(f"{'='*60}")
    print(f"GPU 0 append time:     {append_gpu0_time_ms:.3f} ms")
    print(f"GPU 0 -> GPU 1 transfer: {transfer_time_ms:.3f} ms")
    print(f"Transfer/Append ratio:  {transfer_time_ms/append_gpu0_time_ms:.3f}x")
    if transfer_time_ms < append_gpu0_time_ms:
        overlap_benefit = append_gpu0_time_ms - transfer_time_ms
        print(f"✅ Transfer completed {overlap_benefit:.1f} ms before append finished")
        print(f"   Transfer is hidden by computation!")
    else:
        extra_time = transfer_time_ms - append_gpu0_time_ms
        print(f"⚠️  Transfer took {extra_time:.1f} ms longer than append")
    print(f"{'='*60}\n")
    
    print(f"\nBefore GPU1 append:")
    print(f"  GPU1 num_global_block: {past_key_value_gpu1.num_global_block}")
    print(f"  GPU1 global_blocks length: {len(past_key_value_gpu1.global_blocks[0])}")
    print(f"  GPU1 global_remainder shape: {past_key_value_gpu1.global_remainder[0].shape}")
    
    with torch.cuda.device(1):
        append_gpu1_start.record()
        torch.cuda.nvtx.range_push("GPU1_Append")
        
        o_gpu1 = past_key_value_gpu1.append(
            local_q_gpu1, local_k_gpu1, local_v_gpu1,
            global_q_gpu1, global_k_gpu1, global_v_gpu1,
        )
        
        torch.cuda.nvtx.range_pop()
        append_gpu1_end.record()
    
    print(f"\nAfter GPU1 append:")
    print(f"  GPU1 num_global_block: {past_key_value_gpu1.num_global_block}")
    print(f"  GPU1 global_blocks length: {len(past_key_value_gpu1.global_blocks[0])}")
    
    # Synchronize both GPUs
    torch.cuda.synchronize(device=0)
    torch.cuda.synchronize(device=1)
    
    append_gpu1_time_ms = append_gpu1_start.elapsed_time(append_gpu1_end)
    
    print(f"GPU 1 append time:     {append_gpu1_time_ms:.3f} ms")
    print(f"\n{'='*60}")
    print(f"Pipeline Summary:")
    print(f"{'='*60}")
    print(f"GPU 0 append:          {append_gpu0_time_ms:.3f} ms")
    print(f"Transfer (overlapped): {transfer_time_ms:.3f} ms")
    print(f"GPU 1 append:          {append_gpu1_time_ms:.3f} ms")
    print(f"Total pipeline time:   {max(append_gpu0_time_ms, transfer_time_ms) + append_gpu1_time_ms:.3f} ms")
    print(f"Sequential time would be: {append_gpu0_time_ms + transfer_time_ms + append_gpu1_time_ms:.3f} ms")
    print(f"Time saved by overlap: {min(append_gpu0_time_ms, transfer_time_ms):.3f} ms")
    print(f"{'='*60}\n")
    
    print(f"o_gpu0 shape: {o_gpu0.shape}")
    print(f"o_gpu1 shape: {o_gpu1.shape}")
    print("Pipeline test completed successfully!")

if __name__ == "__main__":
    __main__()
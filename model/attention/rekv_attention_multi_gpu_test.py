import torch
import threading
import os

from .kv_cache_manager import ContextManager
from .position_bias_utils import load_position_bias

def __main__():
    # Load position_bias for GPU 0
    position_bias_gpu0 = load_position_bias("/root/mwnoh/ReKV/model/attention/position_bias.pkl", device="cuda:0")
    # Load position_bias for GPU 1  
    position_bias_gpu1 = load_position_bias("/root/mwnoh/ReKV/model/attention/position_bias.pkl", device="cuda:1")
    # Load position_bias for GPU 2  
    position_bias_gpu2 = load_position_bias("/root/mwnoh/ReKV/model/attention/position_bias.pkl", device="cuda:2")
    
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
    past_key_value_gpu2 = ContextManager(
        position_bias_gpu2,
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
    
    # Flag to control PCIe bandwidth contention
    # True: Enable overlap (Stage 1 transfer during append) - causes PCIe contention with GPU->CPU offloading
    # False: Disable overlap (Stage 1 skipped, all transfer in Stage 2) - no contention but longer pipeline
    ENABLE_OVERLAP = os.environ.get('ENABLE_OVERLAP', 'False').lower() in ('true', '1', 'yes')
    print(f"\n{'='*60}")
    print(f"ENABLE_OVERLAP = {ENABLE_OVERLAP}")
    print(f"{'='*60}\n")
    
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

    h_q_gpu2 = torch.randn(batch_size, num_heads, len_q, dim_head, dtype=torch.float16, device="cuda:2")
    h_k_gpu2 = torch.randn(batch_size, num_heads, len_q, dim_head, dtype=torch.float16, device="cuda:2")
    h_v_gpu2 = torch.randn(batch_size, num_heads, len_q, dim_head, dtype=torch.float16, device="cuda:2")

    local_q_gpu2, local_k_gpu2, local_v_gpu2 = h_q_gpu2, h_k_gpu2, h_v_gpu2
    global_q_gpu2, global_k_gpu2, global_v_gpu2 = h_q_gpu2, h_k_gpu2, h_v_gpu2  

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

    if not past_key_value_gpu2.initialized:
        past_key_value_gpu2.init(local_q_gpu2, local_k_gpu2, local_v_gpu2, 
                                 global_q_gpu2, global_k_gpu2, global_v_gpu2)

    h_q_gpu0 = torch.randn(batch_size, num_heads, len_q, dim_head, dtype=torch.float16, device="cuda:0")
    h_k_gpu0 = torch.randn(batch_size, num_heads, len_q, dim_head, dtype=torch.float16, device="cuda:0")
    h_v_gpu0 = torch.randn(batch_size, num_heads, len_q, dim_head, dtype=torch.float16, device="cuda:0")

    local_q_gpu0, local_k_gpu0, local_v_gpu0 = h_q_gpu0, h_k_gpu0, h_v_gpu0
    global_q_gpu0, global_k_gpu0, global_v_gpu0 = h_q_gpu0, h_k_gpu0, h_v_gpu0


    # Create CUDA streams - separate transfer and compute streams
    gpu0_compute_stream = torch.cuda.Stream(device=0)
    gpu1_transfer_stream = torch.cuda.Stream(device=1)
    gpu1_compute_stream = torch.cuda.Stream(device=1)
    gpu2_transfer_stream = torch.cuda.Stream(device=2)
    gpu2_compute_stream = torch.cuda.Stream(device=2)
    
    # Events for timing
    append_gpu0_start = torch.cuda.Event(enable_timing=True)
    append_gpu0_end = torch.cuda.Event(enable_timing=True)
    transfer_gpu0_to_gpu1_start = torch.cuda.Event(enable_timing=True)
    transfer_gpu0_to_gpu1_end = torch.cuda.Event(enable_timing=True)
    append_gpu1_start = torch.cuda.Event(enable_timing=True)
    append_gpu1_end = torch.cuda.Event(enable_timing=True)
    transfer_gpu1_to_gpu2_start = torch.cuda.Event(enable_timing=True)
    transfer_gpu1_to_gpu2_end = torch.cuda.Event(enable_timing=True)
    append_gpu2_start = torch.cuda.Event(enable_timing=True)
    append_gpu2_end = torch.cuda.Event(enable_timing=True)
    
    # Shared flags for synchronization
    # GPU0 -> GPU1
    transfer_gpu0_to_gpu1_ready = threading.Event()
    gpu0_append_complete = threading.Event()
    transfer_gpu0_to_gpu1_complete = threading.Event()
    
    # GPU1 -> GPU2
    transfer_gpu1_to_gpu2_ready = threading.Event()
    gpu1_append_complete = threading.Event()
    transfer_gpu1_to_gpu2_complete = threading.Event()
    
    # Synchronize before starting
    torch.cuda.synchronize(device=0)
    torch.cuda.synchronize(device=1)
    torch.cuda.synchronize(device=2)
    
    # Define transfer function to run in parallel thread
    def transfer_to_gpu1():
        """Transfer KV cache from GPU 0 to GPU 1 in two stages:
        Stage 1: Transfer input tensors (local_k_gpu0, local_v_gpu0) immediately when append starts
        Stage 2: Transfer all updated states after append completes
        """
        # Stage 1: Transfer input tensors as soon as append starts
        transfer_gpu0_to_gpu1_ready.wait()
        
        with torch.cuda.device(1):
            with torch.cuda.stream(gpu1_transfer_stream):
                transfer_gpu0_to_gpu1_start.record(stream=gpu1_transfer_stream)
                
        if ENABLE_OVERLAP:
            print("Stage 1: Transferring input tensors (local_k_gpu0, local_v_gpu0) to GPU 1 [OVERLAP ENABLED]...")
            with torch.cuda.device(1):
                with torch.cuda.stream(gpu1_transfer_stream):
                    torch.cuda.nvtx.range_push("GPU0_to_GPU1_Stage1_Transfer")
                    
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
        else:
            print("Stage 1: SKIPPED [OVERLAP DISABLED - avoiding PCIe contention]")
        
        # Stage 2: Wait for GPU0 append to complete, then transfer all updated states
        gpu0_append_complete.wait()
        
        if ENABLE_OVERLAP:
            print("Stage 2: Transferring updated states after GPU 0 append completion...")
        else:
            print("Stage 2: Transferring ALL states after GPU 0 append completion (including local_k/v)...")
        
        with torch.cuda.device(1):
            with torch.cuda.stream(gpu1_transfer_stream):
                torch.cuda.nvtx.range_push("GPU0_to_GPU1_State_Transfer")
                
                # Transfer input tensors local_k_gpu0 and local_v_gpu0 to GPU1 if not done in Stage 1
                if not ENABLE_OVERLAP:
                    torch.cuda.nvtx.range_push("no overlap kv cache transfer")
                    local_k_gpu1_new = local_k_gpu0.to("cuda:1", non_blocking=True)
                    local_v_gpu1_new = local_v_gpu0.to("cuda:1", non_blocking=True)
                    torch.cuda.nvtx.range_pop()

                    torch.cuda.nvtx.range_push("kv cache torch cat")
                    past_key_value_gpu1.local_k = torch.cat((past_key_value_gpu1.local_k, local_k_gpu1_new), dim=-2)
                    past_key_value_gpu1.local_v = torch.cat((past_key_value_gpu1.local_v, local_v_gpu1_new), dim=-2)
                    torch.cuda.nvtx.range_pop()

                    # Apply n_local restriction (same as in ContextManager.append)
                    if past_key_value_gpu1.local_k.size(-2) >= past_key_value_gpu1.n_local:
                        past_key_value_gpu1.local_k = past_key_value_gpu1.local_k[:, :, -past_key_value_gpu1.n_local:, :]
                        past_key_value_gpu1.local_v = past_key_value_gpu1.local_v[:, :, -past_key_value_gpu1.n_local:, :]
                
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
                past_key_value_gpu1.global_blocks = past_key_value_gpu0.global_blocks
                
                # Copy block_k data (representative keys) from GPU0 to GPU1
                for u in range(past_key_value_gpu0.num_units):
                    if past_key_value_gpu0.block_k[u].length > 0:
                        gpu0_data = past_key_value_gpu0.block_k[u].get_data()  # (length, hidden_size)
                        gpu1_data = gpu0_data.to("cuda:1", non_blocking=True)
                        past_key_value_gpu1.block_k[u].append(gpu1_data)  # Append all at once
                
                torch.cuda.nvtx.range_pop()
                transfer_gpu0_to_gpu1_end.record(stream=gpu1_transfer_stream)
        
        transfer_gpu0_to_gpu1_complete.set()
    
    def transfer_to_gpu2():
        """Transfer KV cache from GPU 1 to GPU 2 in two stages:
        Stage 1: Transfer input tensors (local_k_gpu0, local_v_gpu0) immediately when GPU1 append starts
        Stage 2: Transfer all updated states after GPU1 append completes
        """
        # Stage 1: Transfer input tensors as soon as GPU1 append starts
        transfer_gpu1_to_gpu2_ready.wait()
        
        with torch.cuda.device(2):
            with torch.cuda.stream(gpu2_transfer_stream):
                transfer_gpu1_to_gpu2_start.record(stream=gpu2_transfer_stream)
                
        if ENABLE_OVERLAP:
            print("Stage 1: Transferring input tensors (local_k_gpu1, local_v_gpu1) to GPU 2 [OVERLAP ENABLED]...")
            with torch.cuda.device(2):
                with torch.cuda.stream(gpu2_transfer_stream):
                    torch.cuda.nvtx.range_push("GPU1_to_GPU2_Stage1_Transfer")
                    
                    # Transfer input tensors local_k_gpu1 and local_v_gpu1 to GPU2
                    # Concatenate with existing local_k and local_v
                    local_k_gpu2_new = local_k_gpu1.to("cuda:2", non_blocking=True)
                    local_v_gpu2_new = local_v_gpu1.to("cuda:2", non_blocking=True)

                    past_key_value_gpu2.local_k = torch.cat((past_key_value_gpu2.local_k, local_k_gpu2_new), dim=-2)
                    past_key_value_gpu2.local_v = torch.cat((past_key_value_gpu2.local_v, local_v_gpu2_new), dim=-2)

                    # Apply n_local restriction (same as in ContextManager.append)
                    if past_key_value_gpu2.local_k.size(-2) >= past_key_value_gpu2.n_local:
                        past_key_value_gpu2.local_k = past_key_value_gpu2.local_k[:, :, -past_key_value_gpu2.n_local:, :]
                        past_key_value_gpu2.local_v = past_key_value_gpu2.local_v[:, :, -past_key_value_gpu2.n_local:, :]
                    
                    torch.cuda.nvtx.range_pop()
        else:
            print("Stage 1: SKIPPED [OVERLAP DISABLED - avoiding PCIe contention]")
        
        # Stage 2: Wait for GPU1 append to complete, then transfer all updated states
        gpu1_append_complete.wait()
        
        if ENABLE_OVERLAP:
            print("Stage 2: Transferring updated states after GPU 1 append completion...")
        else:
            print("Stage 2: Transferring ALL states after GPU 1 append completion (including local_k/v)...")
        
        with torch.cuda.device(2):
            with torch.cuda.stream(gpu2_transfer_stream):
                torch.cuda.nvtx.range_push("GPU1_to_GPU2_State_Transfer")
                
                # Transfer input tensors local_k_gpu0 and local_v_gpu0 to GPU2 if not done in Stage 1
                if not ENABLE_OVERLAP:
                    torch.cuda.nvtx.range_push("no overlap kv cache transfer")
                    local_k_gpu2_new = local_k_gpu1.to("cuda:2", non_blocking=True)
                    local_v_gpu2_new = local_v_gpu1.to("cuda:2", non_blocking=True)
                    torch.cuda.nvtx.range_pop()

                    torch.cuda.nvtx.range_push("kv cache torch cat")
                    past_key_value_gpu2.local_k = torch.cat((past_key_value_gpu2.local_k, local_k_gpu2_new), dim=-2)
                    past_key_value_gpu2.local_v = torch.cat((past_key_value_gpu2.local_v, local_v_gpu2_new), dim=-2)
                    torch.cuda.nvtx.range_pop()
                    # Apply n_local restriction (same as in ContextManager.append)
                    if past_key_value_gpu2.local_k.size(-2) >= past_key_value_gpu2.n_local:
                        past_key_value_gpu2.local_k = past_key_value_gpu2.local_k[:, :, -past_key_value_gpu2.n_local:, :]
                        past_key_value_gpu2.local_v = past_key_value_gpu2.local_v[:, :, -past_key_value_gpu2.n_local:, :]
                
                # Copy updated states from GPU1's append
                
                # Copy global remainder (updated during append)
                past_key_value_gpu2.global_remainder = (
                    past_key_value_gpu1.global_remainder[0].to("cuda:2", non_blocking=True),
                    past_key_value_gpu1.global_remainder[1].to("cuda:2", non_blocking=True),
                )
                past_key_value_gpu2._global_remainder_st = past_key_value_gpu1._global_remainder_st
                past_key_value_gpu2._global_remainder_ed = past_key_value_gpu1._global_remainder_ed
                
                # Copy init KV (may be updated during append)
                past_key_value_gpu2.init_k = past_key_value_gpu1.init_k.to("cuda:2", non_blocking=True)
                past_key_value_gpu2.init_v = past_key_value_gpu1.init_v.to("cuda:2", non_blocking=True)
                past_key_value_gpu2.init_exc = past_key_value_gpu1.init_exc
                
                # Copy length (updated during append)
                past_key_value_gpu2.length = past_key_value_gpu1.length
                
                # Copy num_global_block (increases when new blocks are created during append)
                past_key_value_gpu2.num_global_block = past_key_value_gpu1.num_global_block
                
                # Copy global_blocks (CPU-based, can be shared)
                past_key_value_gpu2.global_blocks = past_key_value_gpu1.global_blocks
                
                # Copy block_k data (representative keys) from GPU1 to GPU2
                for u in range(past_key_value_gpu1.num_units):
                    if past_key_value_gpu1.block_k[u].length > 0:
                        gpu1_data = past_key_value_gpu1.block_k[u].get_data()  # (length, hidden_size)
                        gpu2_data = gpu1_data.to("cuda:2", non_blocking=True)
                        past_key_value_gpu2.block_k[u].append(gpu2_data)  # Append all at once
                
                torch.cuda.nvtx.range_pop()
                transfer_gpu1_to_gpu2_end.record(stream=gpu2_transfer_stream)
        
        transfer_gpu1_to_gpu2_complete.set()
    
    # Start transfer threads
    transfer_thread_gpu0_to_gpu1 = threading.Thread(target=transfer_to_gpu1)
    transfer_thread_gpu1_to_gpu2 = threading.Thread(target=transfer_to_gpu2)
    transfer_thread_gpu0_to_gpu1.start()
    transfer_thread_gpu1_to_gpu2.start()
    
    # Process batch on GPU 0
    print("\n" + "="*60)
    print("GPU 0 Processing")
    print("="*60)
    print(f"Before GPU0 append (second batch):")
    print(f"  GPU0 num_global_block: {past_key_value_gpu0.num_global_block}")
    print(f"  GPU0 global_blocks length: {len(past_key_value_gpu0.global_blocks[0])}")
    print(f"  GPU0 global_remainder shape: {past_key_value_gpu0.global_remainder[0].shape}")
    
    # Signal transfer thread to start Stage 1 (input tensor transfer) immediately
    print("Signaling GPU0->GPU1 transfer thread to begin Stage 1 (input tensors)...")
    transfer_gpu0_to_gpu1_ready.set()
    
    with torch.cuda.device(0):
        with torch.cuda.stream(gpu0_compute_stream):
            append_gpu0_start.record(stream=gpu0_compute_stream)
            if ENABLE_OVERLAP:
                torch.cuda.nvtx.range_push("GPU0_Append_Overlapped")
            else:
                torch.cuda.nvtx.range_push("GPU0_Append_No_Overlap")
            
            o_gpu0 = past_key_value_gpu0.append(
                local_q_gpu0, local_k_gpu0, local_v_gpu0,
                global_q_gpu0, global_k_gpu0, global_v_gpu0,
            )
            
            torch.cuda.nvtx.range_pop()
            append_gpu0_end.record(stream=gpu0_compute_stream)
    
    print(f"After GPU0 append (second batch):")
    print(f"  GPU0 num_global_block: {past_key_value_gpu0.num_global_block}")
    print(f"  GPU0 global_blocks length: {len(past_key_value_gpu0.global_blocks[0])}")
    
    # Signal that GPU0 append is complete - start Stage 2 transfer
    print("GPU 0 append complete - signaling Stage 2 transfer...")
    gpu0_append_complete.set()
    
    # Wait for GPU0->GPU1 transfer to complete
    print("Waiting for GPU0->GPU1 transfer to complete...")
    transfer_thread_gpu0_to_gpu1.join()
    torch.cuda.synchronize(device=0)
    torch.cuda.synchronize(device=1)
    
    # Calculate GPU0 timing
    append_gpu0_time_ms = append_gpu0_start.elapsed_time(append_gpu0_end)
    transfer_gpu0_to_gpu1_time_ms = transfer_gpu0_to_gpu1_start.elapsed_time(transfer_gpu0_to_gpu1_end)
    
    print(f"\n{'='*60}")
    print(f"GPU 0 Results:")
    print(f"{'='*60}")
    print(f"GPU 0 append time:          {append_gpu0_time_ms:.3f} ms")
    print(f"GPU 0 -> GPU 1 transfer:    {transfer_gpu0_to_gpu1_time_ms:.3f} ms")
    print(f"Transfer/Append ratio:      {transfer_gpu0_to_gpu1_time_ms/append_gpu0_time_ms:.3f}x")
    if transfer_gpu0_to_gpu1_time_ms < append_gpu0_time_ms:
        overlap_benefit = append_gpu0_time_ms - transfer_gpu0_to_gpu1_time_ms
        print(f"✅ Transfer completed {overlap_benefit:.1f} ms before append finished")
        print(f"   Transfer is hidden by computation!")
    else:
        extra_time = transfer_gpu0_to_gpu1_time_ms - append_gpu0_time_ms
        print(f"⚠️  Transfer took {extra_time:.1f} ms longer than append")
    print(f"{'='*60}\n")
    
    # Process batch on GPU 1 (with overlap to GPU 2)
    print("\n" + "="*60)
    print("GPU 1 Processing")
    print("="*60)
    print(f"Before GPU1 append:")
    print(f"  GPU1 num_global_block: {past_key_value_gpu1.num_global_block}")
    print(f"  GPU1 global_blocks length: {len(past_key_value_gpu1.global_blocks[0])}")
    print(f"  GPU1 global_remainder shape: {past_key_value_gpu1.global_remainder[0].shape}")
    
    # Signal transfer thread to start GPU1->GPU2 transfer
    print("Signaling GPU1->GPU2 transfer thread to begin Stage 1 (input tensors)...")
    transfer_gpu1_to_gpu2_ready.set()
    
    with torch.cuda.device(1):
        with torch.cuda.stream(gpu1_compute_stream):
            append_gpu1_start.record(stream=gpu1_compute_stream)
            if ENABLE_OVERLAP:
                torch.cuda.nvtx.range_push("GPU1_Append_Overlapped")
            else:
                torch.cuda.nvtx.range_push("GPU1_Append_No_Overlap")
            
            o_gpu1 = past_key_value_gpu1.append(
                local_q_gpu1, local_k_gpu1, local_v_gpu1,
                global_q_gpu1, global_k_gpu1, global_v_gpu1,
            )
            
            torch.cuda.nvtx.range_pop()
            append_gpu1_end.record(stream=gpu1_compute_stream)
    
    print(f"After GPU1 append:")
    print(f"  GPU1 num_global_block: {past_key_value_gpu1.num_global_block}")
    print(f"  GPU1 global_blocks length: {len(past_key_value_gpu1.global_blocks[0])}")
    
    # Signal that GPU1 append is complete
    print("GPU 1 append complete - signaling Stage 2 transfer...")
    gpu1_append_complete.set()
    
    # Wait for GPU1->GPU2 transfer to complete
    print("Waiting for GPU1->GPU2 transfer to complete...")
    transfer_thread_gpu1_to_gpu2.join()
    torch.cuda.synchronize(device=1)
    torch.cuda.synchronize(device=2)
    
    # Calculate GPU1 timing
    append_gpu1_time_ms = append_gpu1_start.elapsed_time(append_gpu1_end)
    transfer_gpu1_to_gpu2_time_ms = transfer_gpu1_to_gpu2_start.elapsed_time(transfer_gpu1_to_gpu2_end)
    
    print(f"\n{'='*60}")
    print(f"GPU 1 Results:")
    print(f"{'='*60}")
    print(f"GPU 1 append time:          {append_gpu1_time_ms:.3f} ms")
    print(f"GPU 1 -> GPU 2 transfer:    {transfer_gpu1_to_gpu2_time_ms:.3f} ms")
    print(f"Transfer/Append ratio:      {transfer_gpu1_to_gpu2_time_ms/append_gpu1_time_ms:.3f}x")
    if transfer_gpu1_to_gpu2_time_ms < append_gpu1_time_ms:
        overlap_benefit = append_gpu1_time_ms - transfer_gpu1_to_gpu2_time_ms
        print(f"✅ Transfer completed {overlap_benefit:.1f} ms before append finished")
        print(f"   Transfer is hidden by computation!")
    else:
        extra_time = transfer_gpu1_to_gpu2_time_ms - append_gpu1_time_ms
        print(f"⚠️  Transfer took {extra_time:.1f} ms longer than append")
    print(f"{'='*60}\n")
    
    # Process batch on GPU 2
    print("\n" + "="*60)
    print("GPU 2 Processing")
    print("="*60)
    print(f"Before GPU2 append:")
    print(f"  GPU2 num_global_block: {past_key_value_gpu2.num_global_block}")
    print(f"  GPU2 global_blocks length: {len(past_key_value_gpu2.global_blocks[0])}")
    print(f"  GPU2 global_remainder shape: {past_key_value_gpu2.global_remainder[0].shape}")
    
    with torch.cuda.device(2):
        with torch.cuda.stream(gpu2_compute_stream):
            append_gpu2_start.record(stream=gpu2_compute_stream)
            torch.cuda.nvtx.range_push("GPU2_Append")
            
            o_gpu2 = past_key_value_gpu2.append(
                local_q_gpu2, local_k_gpu2, local_v_gpu2,
                global_q_gpu2, global_k_gpu2, global_v_gpu2,
            )
            
            torch.cuda.nvtx.range_pop()
            append_gpu2_end.record(stream=gpu2_compute_stream)
    
    print(f"After GPU2 append:")
    print(f"  GPU2 num_global_block: {past_key_value_gpu2.num_global_block}")
    print(f"  GPU2 global_blocks length: {len(past_key_value_gpu2.global_blocks[0])}")
    
    # Synchronize all GPUs
    torch.cuda.synchronize(device=0)
    torch.cuda.synchronize(device=1)
    torch.cuda.synchronize(device=2)
    
    append_gpu2_time_ms = append_gpu2_start.elapsed_time(append_gpu2_end)
    
    print(f"\n{'='*60}")
    print(f"GPU 2 Results:")
    print(f"{'='*60}")
    print(f"GPU 2 append time:          {append_gpu2_time_ms:.3f} ms")
    print(f"{'='*60}\n")
    
    print(f"\n{'='*60}")
    if ENABLE_OVERLAP:
        print(f"(ENABLE_OVERLAP=True):")
    else:
        print(f"(ENABLE_OVERLAP=False):")
    print(f"{'='*60}")
    print(f"GPU 0 append:                    {append_gpu0_time_ms:.3f} ms")
    print(f"GPU 0 -> GPU 1 transfer:         {transfer_gpu0_to_gpu1_time_ms:.3f} ms")
    print(f"GPU 1 append:                    {append_gpu1_time_ms:.3f} ms")
    print(f"GPU 1 -> GPU 2 transfer:         {transfer_gpu1_to_gpu2_time_ms:.3f} ms")
    print(f"GPU 2 append:                    {append_gpu2_time_ms:.3f} ms")
    print(f"")
    if ENABLE_OVERLAP:
        stage1_time = max(append_gpu0_time_ms, transfer_gpu0_to_gpu1_time_ms)
        stage2_time = max(append_gpu1_time_ms, transfer_gpu1_to_gpu2_time_ms)
        stage3_time = append_gpu2_time_ms
    else:
        stage1_time = append_gpu0_time_ms + transfer_gpu0_to_gpu1_time_ms
        stage2_time = append_gpu1_time_ms + transfer_gpu1_to_gpu2_time_ms
        stage3_time = append_gpu2_time_ms
    total_time = stage1_time + stage2_time + stage3_time
    
    print(f"Total time:             {total_time:.3f} ms")
    print(f"  Stage 1 (GPU0 + transfer):     {stage1_time:.3f} ms")
    print(f"  Stage 2 (GPU1 + transfer):     {stage2_time:.3f} ms")
    print(f"  Stage 3 (GPU2):                {stage3_time:.3f} ms")
    print(f"")
    print(f"{'='*60}\n")
    
    print(f"o_gpu0 shape: {o_gpu0.shape}")
    print(f"o_gpu1 shape: {o_gpu1.shape}")
    print(f"o_gpu2 shape: {o_gpu2.shape}")
    print("3-GPU Pipeline test completed successfully!")

if __name__ == "__main__":
    __main__()
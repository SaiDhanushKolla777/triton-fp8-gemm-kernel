import torch
from task import input_t, output_t
from triton_kernel import triton_fp8_matmul
"""
def custom_kernel(data: input_t) -> output_t:
    """
    Optimized implementation of block-scale fp8 gemm
    Args:
        data: Tuple that expands to:
            a: torch.Tensor[float8_e4m3fnuz] of shape [m, k], 
            b: torch.Tensor[float8_e4m3fnuz] of shape [n, k], 
            a_scale: torch.Tensor[float32] of shape [m, k // 128], 
            b_scale: torch.Tensor[float32] of shape [n // 128, k // 128], 
            c: torch.Tensor[bfloat16] of shape [m, n]
    Returns:
        Tensor containing output in bf16
    """
    # Extract the tensors
    a, b, a_scale, b_scale, c = data
    
    # Ensure tensors are contiguous for better memory access
    a = a.contiguous()
    b = b.contiguous()
    a_scale = a_scale.contiguous()
    b_scale = b_scale.contiguous()
    
    # Constants
    m = a.shape[0]
    n = b.shape[0]
    k = a.shape[1]
    block_k = 128
    block_n = 128
    
    # Initialize accumulator in BF16 for better performance
    accum = torch.zeros((m, n), dtype=torch.bfloat16, device=a.device)
    
    # Process in blocks to match the scale factor structure
    for k_idx in range(0, k, block_k):
        k_block_idx = k_idx // block_k
        
        # Convert current blocks to BF16 to save memory
        a_block = a[:, k_idx:k_idx+block_k].to(torch.bfloat16)
        
        # Get a_scale for current k block and expand for broadcasting
        a_scale_block = a_scale[:, k_block_idx].unsqueeze(1).to(torch.bfloat16)
        
        # Apply a_scale to current block
        a_scaled = a_block * a_scale_block
        
        # Process b in blocks to match n-dimension scaling
        for n_idx in range(0, n, block_n):
            n_end = min(n_idx + block_n, n)
            n_block_idx = n_idx // block_n
            
            # Get current b block and convert to BF16
            b_block = b[n_idx:n_end, k_idx:k_idx+block_k].to(torch.bfloat16)
            
            # Get and apply b_scale
            b_scale_val = b_scale[n_block_idx, k_block_idx].to(torch.bfloat16)
            b_scaled = b_block * b_scale_val
            
            # Compute and accumulate partial product
            accum[:, n_idx:n_end] += torch.matmul(a_scaled, b_scaled.T)
    
    # Copy result to output tensor
    c.copy_(accum)
    
    return c
"""
def custom_kernel(data: input_t) -> output_t:
    a, b, a_scale, b_scale, c = data
    return triton_fp8_matmul(a, b, a_scale, b_scale, c)


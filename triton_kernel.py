import triton
import triton.language as tl
import torch

# Block sizes (must divide K exactly since K % 128 == 0 by problem statement)
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 128


@triton.jit
def fp8_blockwise_gemm(
    A, B, C, a_scale, b_scale,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    stride_asm, stride_ask,
    stride_bsn, stride_bsk,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        # -----------------------
        # Load A tile [BLOCK_M, BLOCK_K]
        # -----------------------
        a_ptrs = A + (offs_m[:, None] * stride_am + (k0 + offs_k)[None, :] * stride_ak)
        a_tile = tl.load(a_ptrs, mask=offs_m[:, None] < M, other=0).to(tl.float32)

        # Apply A scaling: each (m, k_blk) has scale a_scale[m, k_blk]
        k_blk = k0 // BLOCK_K
        a_scale_ptrs = a_scale + offs_m * stride_asm + k_blk * stride_ask
        scale_a = tl.load(a_scale_ptrs, mask=offs_m < M, other=1.0)
        a_tile = a_tile * scale_a[:, None]

        # -----------------------
        # Load B tile [BLOCK_N, BLOCK_K]
        # -----------------------
        b_ptrs = B + (offs_n[:, None] * stride_bn + (k0 + offs_k)[None, :] * stride_bk)
        b_tile = tl.load(b_ptrs, mask=offs_n[:, None] < N, other=0).to(tl.float32)

        # Apply B scaling: each (n_blk, k_blk) has scale b_scale[n_blk, k_blk]
        n_blk = offs_n // BLOCK_N
        b_scale_ptrs = b_scale + n_blk[:, None] * stride_bsn + k_blk * stride_bsk
        scale_b = tl.load(b_scale_ptrs, mask=offs_n[:, None] < N, other=1.0)
        b_tile = b_tile * scale_b

        # -----------------------
        # Accumulate
        # -----------------------
        acc += tl.dot(a_tile, tl.trans(b_tile))

    # -----------------------
    # Store C in BF16
    # -----------------------
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=mask)


def triton_fp8_matmul(a: torch.Tensor,
                      b: torch.Tensor,
                      a_scale: torch.Tensor,
                      b_scale: torch.Tensor,
                      c: torch.Tensor):
    """
    Triton implementation of block-scaled FP8 GEMM.
    Args:
        a: [M, K] FP8 column-major
        b: [N, K] FP8 column-major
        a_scale: [M, K//128] FP32
        b_scale: [N//128, K//128] FP32
        c: [M, N] BF16 row-major (preallocated)
    Returns:
        c: [M, N] result in BF16
    """
    M, K = a.shape
    N = b.shape[0]

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    fp8_blockwise_gemm[grid](
        a, b, c, a_scale, b_scale,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        a_scale.stride(0), a_scale.stride(1),
        b_scale.stride(0), b_scale.stride(1),
    )
    return c

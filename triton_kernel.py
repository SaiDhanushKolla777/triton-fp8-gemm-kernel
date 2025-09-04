import triton
import triton.language as tl
import torch

# Default tile sizes; K scaling is per 128 so we keep BLOCK_K=128
DEFAULT_BLOCK_M = 128
DEFAULT_BLOCK_N = 128
DEFAULT_BLOCK_K = 128

CONFIGS = [
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128},
        num_warps=8, num_stages=4
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128},
        num_warps=4, num_stages=5
    ),
]

@triton.autotune(configs=CONFIGS, key=["M", "N", "K"])
@triton.jit
def fp8_blockwise_gemm(
    A, B, C, a_scale, b_scale,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    stride_asm, stride_ask,
    stride_bsn, stride_bsk,
    # constexprs for compile-time specialization
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D launch: one program per (tile_m, tile_n)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Accumulator in FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate K in 128-chunks; unroll/pipeline
    tl.static_assert(BLOCK_K == 128, "BLOCK_K must be 128 to match scaling granularity")
    for k0 in tl.static_range(0,  K,  BLOCK_K):
        # ---- Load A tile [BLOCK_M, BLOCK_K] (column-major strides) ----
        a_ptrs = A + (offs_m[:, None] * stride_am + (k0 + offs_k)[None, :] * stride_ak)
        a_tile_fp8 = tl.load(a_ptrs, mask=offs_m[:, None] < M, other=0)

        # ---- Apply A scaling (per row, per k-block) ----
        k_blk = k0 // BLOCK_K
        a_scale_ptrs = a_scale + offs_m * stride_asm + k_blk * stride_ask
        scale_a = tl.load(a_scale_ptrs, mask=offs_m < M, other=1.0)

        # Convert to BF16 *after* scaling to hit MFMA (acc stays FP32)
        a_tile = (a_tile_fp8.to(tl.float32) * scale_a[:, None]).to(tl.bfloat16)

        # ---- Load B tile [BLOCK_N, BLOCK_K] (column-major strides) ----
        b_ptrs = B + (offs_n[:, None] * stride_bn + (k0 + offs_k)[None, :] * stride_bk)
        b_tile_fp8 = tl.load(b_ptrs, mask=offs_n[:, None] < N, other=0)

        # ---- Apply B scaling (per [n_blk, k_blk]) ----
        n_blk = offs_n // BLOCK_N
        b_scale_ptrs = b_scale + n_blk[:, None] * stride_bsn + k_blk * stride_bsk
        scale_b = tl.load(b_scale_ptrs, mask=offs_n[:, None] < N, other=1.0)

        b_tile = (b_tile_fp8.to(tl.float32) * scale_b).to(tl.bfloat16)

        # ---- FMA: bf16 x bf16 -> fp32 ----
        acc += tl.dot(a_tile, tl.trans(b_tile))

    # ---- Store C as BF16 (row-major) ----
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
    a: [M, K] FP8 e4m3fnuz, column-major
    b: [N, K] FP8 e4m3fnuz, column-major
    a_scale: [M, K//128] FP32
    b_scale: [N//128, K//128] FP32
    c: [M, N] BF16 row-major (preallocated)
    """
    M, K = a.shape
    N = b.shape[0]
    # K divisible by 128 guaranteed by the task
    assert (K % DEFAULT_BLOCK_K) == 0, "K must be a multiple of 128"

    grid = (triton.cdiv(M, DEFAULT_BLOCK_M), triton.cdiv(N, DEFAULT_BLOCK_N))

    fp8_blockwise_gemm[grid](
        a, b, c, a_scale, b_scale,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        a_scale.stride(0), a_scale.stride(1),
        b_scale.stride(0), b_scale.stride(1),
        BLOCK_M=DEFAULT_BLOCK_M,
        BLOCK_N=DEFAULT_BLOCK_N,
        BLOCK_K=DEFAULT_BLOCK_K,
    )
    return c

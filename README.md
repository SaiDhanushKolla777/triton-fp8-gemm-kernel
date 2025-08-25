
# Triton FP8 Groupwise GEMM Kernel

This repository contains an implementation of **blockwise FP8 GEMM with scaling factors**, built for the **AMD Inference Sprint 2025** challenge.  
The project explores different ways to implement the kernel:
- A **reference version** (slow, always correct),
- A **Triton prototype** (good for learning and mid-tier performance),
- A **HIP template** (intended for the fastest implementation on MI300 GPUs).

---

## 1. Problem Description

We need to multiply two matrices stored in **FP8** (`float8_e4m3fnuz`) with **blockwise scaling factors**, and accumulate the result in **BF16**.

- Input matrices are **column-major**.
- Output matrix is **row-major**.
- Scaling is **not per element**:
  - **`a_scale`**: Each row of `a` has a scale per **128-column block**.
  - **`b_scale`**: Each `[128 × 128]` block of `b` shares one scale.

The final result is:

```

c = (a \* a\_scale) @ (b \* b\_scale).T

```

with accumulation in FP32, then cast to BF16.

---

## 2. File Overview

| File              | What it does |
|-------------------|--------------|
| `reference.py`    | Inefficient baseline. Expands scales, dequantizes, then calls `torch.matmul`. |
| `submission.py`   | Main entry point. The competition harness calls `custom_kernel` from this file. |
| `triton_kernel.py`| Our Triton kernel. Applies scaling inside the matmul loop, accumulates in FP32, stores BF16. |
| `template.py`     | Minimal Python template. Mostly for teaching, not for performance. |
| `template-hip.py` | HIP C++ template. Intended for the fastest MI300 solution. |
| `task.py`         | Type definitions for inputs/outputs. |
| `task.yml`        | Defines test shapes and benchmark configs. |
| `utils.py`        | Helpers (e.g. compare output with reference). |
| `eval.py`         | Script that runs tests and benchmarks. |

---

## 3. Input/Output Details

### Input Tensors
- **`a`**: `[M, K]`, FP8, column-major  
- **`b`**: `[N, K]`, FP8, column-major  
- **`a_scale`**: `[M, K//128]`, FP32  
  - `a_scale[m, blk]` applies to `a[m, blk*128 : (blk+1)*128]`.
- **`b_scale`**: `[N//128, K//128]`, FP32  
  - `b_scale[n_blk, k_blk]` applies to the block `b[n_blk*128:(n_blk+1)*128, k_blk*128:(k_blk+1)*128]`.

### Output Tensor
- **`c`**: `[M, N]`, BF16, row-major.  
  Accumulated in FP32, cast to BF16 on store.

---

## 4. ASCII Diagrams of Scaling

### A (LHS) Scaling — `a_scale`
Each row has a different scale for every 128-wide block of columns:

```

a (M x K)

Row m →
+---------+---------+---------+----
\| blk 0   | blk 1   | blk 2   | ...
\| \*s\[m,0] | \*s\[m,1] | \*s\[m,2] |
+---------+---------+---------+----

a\_scale shape: \[M, K//128]

```

So for row `m`, chunk `[0:128]` gets multiplied by `a_scale[m,0]`,  
chunk `[128:256]` by `a_scale[m,1]`, and so on.

---

### B (RHS) Scaling — `b_scale`
Every `[128 × 128]` tile in `b` has its own scale:

```

b (N x K)

K →
0.......128.......256.......384...
N
↓
0  +---------+---------+---------+----
\| blk(0,0)| blk(0,1)| blk(0,2)|
\| \*s\[0,0] | \*s\[0,1] | \*s\[0,2] |
128+---------+---------+---------+----
\| blk(1,0)| blk(1,1)| blk(1,2)|
\| \*s\[1,0] | \*s\[1,1] | \*s\[1,2] |
256+---------+---------+---------+----

````

Here `b_scale[n_blk, k_blk]` multiplies the entire block  
`b[n_blk*128:(n_blk+1)*128, k_blk*128:(k_blk+1)*128]`.

---

## 5. How the Computation Works

1. **Scaling**
   - Apply the correct block scale from `a_scale` and `b_scale` as shown above.
2. **Matmul**
   - Multiply `a_scaled @ b_scaled.T` in FP32.
3. **Result**
   - Convert result to BF16 and write into `c`.

---

## 6. Running Locally

### Install dependencies
```bash
pip install torch triton
````

> On AMD MI300, install ROCm-enabled PyTorch/Triton.

### Test correctness

```python
from reference import generate_input, check_implementation
from submission import custom_kernel
from task import TestSpec

spec = TestSpec(m=128, n=512, k=7168, seed=42)
data = generate_input(**spec)

check_implementation(custom_kernel)(data)
```

### Run benchmarks

```bash
python eval.py
```

---

## 7. Development Workflow

* Start with `reference.py` → see how scaling is applied.
* Explore with `triton_kernel.py` → easier to debug and experiment.
* For max performance: implement in `template-hip.py`.

---

## 8. Performance Hints

* Block size for K = **128** (always guaranteed).
* Apply scaling **inline inside the K-loop**, not by expanding.
* Accumulate in FP32, cast to BF16 once at the end.
* Optimize memory layout and use shared memory for HIP.

---

## 9. Speed of Light (AMD Baselines)

| M    | N    | K    | Time (µs) |
| ---- | ---- | ---- | --------- |
| 1024 | 1536 | 7168 | 8.63      |
| 1024 | 4608 | 7168 | 25.89     |
| 6144 | 1536 | 7168 | 51.78     |
| 6144 | 4608 | 7168 | 155.30    |
| 1024 | 7168 | 256  | 3.17      |
| 6144 | 7168 | 256  | 17.27     |

Your kernel’s score = geometric mean of benchmark ratios vs these times.

---

## 10. Example: Using Triton Kernel

```python
from triton_kernel import triton_fp8_matmul
from reference import generate_input

data = generate_input(m=256, n=512, k=1024, seed=123)
a, b, a_scale, b_scale, c = data

out = triton_fp8_matmul(a, b, a_scale, b_scale, c)
print(out.shape)  # [256, 512]
```

---

## 11. Credits

* AMD & GPUmode for setting up the challenge.
* OpenAI Triton for GPU programming in Python.
* PyTorch team for FP8/BF16 support.

---

## 12. License

MIT License

```


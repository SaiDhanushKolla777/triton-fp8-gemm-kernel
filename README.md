
# Triton FP8 Groupwise GEMM Kernel


This repository provides an implementation for the AMD Inference Sprint 2025 challenge: **Efficient Groupwise GEMM in FP8 with block-wise scaling on MI300 GPUs**.  
It is written to help anyone understand the challenge, the file structure, and how to get started with both correctness and optimization.

---

## 📝 What is the Problem?

- **Task:** Multiply two large matrices stored in **FP8** format (`e4m3fnuz`), scale their blocks using provided scaling factors, and store the result in **BF16** format.
- **Twist:** Scaling isn't per element, but **per block**-so you need to apply each scaling value onto chunks (blocks) of the matrix.
- **Objective:** Do all this as **fast as possible on a GPU** (especially the AMD MI300), while staying correct and matching the expected results.

---

## 📦 Files & What They Do

| File                | Role/Description                                     |
|---------------------|-----------------------------------------------------|
| `README.md`         | **This document.** Explains everything in detail.   |
| `reference.py`      | Organizers’ official reference code. It’s always correct, but not fast. Use it to check your output. |
| `submission.py`     | Where you write your optimized (or just correct) solution. This is your main file for submission. |
| `task.py`           | Data types and shapes for inputs and outputs. Used for automatic validation during testing and benchmarks. |
| `task.yml`          | Describes test cases, benchmarks, and performance goals. Also used by the competition system. |
| `template.py`       | Minimal Python template to help you start from scratch. |
| `template-hip.py`   | Starter template for writing a GPU kernel with HIP (low-level, faster, and more complex). |

---

## 🖼️ Matrix Shapes & Data Explanation

### Inputs

- `a`: Shape `[M, K]`, FP8, **column-major** (i.e., Fortran-style, not numpy-style ordering!)
- `b`: Shape `[N, K]`, FP8, **column-major**
- `a_scale`: `[M, K//128]`, FP32  
  Each value in `a_scale[m, k_blk]` applies to a block of 128 consecutive columns for row `m` in `a`.
- `b_scale`: `[N//128, K//128]`, FP32  
  Each value applies to a `[128,[128]` chunk of `b`.

### Output

- `c`: Shape `[M, N]`, **BF16**, **row-major** (numpy/C style)

---

## 🔄 What Does the Computation Look Like?

1. **Scale the Matrices Blockwise**  
   Multiply each tile/chunk of the input matrices by its corresponding scaling factor (as described above).

2. **Matrix Multiplication**  
   Multiply the block-scaled `a` with the transpose of block-scaled `b`.

3. **Store Result**  
   Write the output into `c` in BF16 (bfloat16) format.

---

## 💡 The Solution Structure (How to Approach It)

### Step 1: Understand Block-wise Scaling

- For `a`:  
  For every row, there’s a scaling factor for every 128 column chunk.  
  Example: For row 0, `a_scale[0,cales columns 0-127, `a_scale[0,` scales 128-255, etc.

- For `b`:  
  For every[128][128] tile (block), there’s a single scaling factor.

### Step 2: Expand the Scaling Factors

- Efficiently broadcast (expand) the scaling factors so they match the shapes of `a` and `b` for multiplication.

### Step 3: Convert Data Types

- FP8 isn't directly supported for computation in PyTorch, so you typically convert FP8 → float32/BF16 for math, then back to BF16 for output.

### Step 4: Matrix Multiply (and Optimize)

- Multiply the block-scaled matrices (`a` with `b.T`) using an efficient method (initially `torch.matmul`, then later custom Triton/HIP kernels for speed).

### Step 5: Output Storage

- Store the result in `c` (row-major, BF16).

---

## 🧑‍💻 Example Code Workflow

Here’s a logical high-level sketch:

```python
# Read inputs: a, b, a_scale, b_scale
# Expand a_scale and b_scale to match the blocks they should scale
a_scaled = ... # multiply/expand a by a_scale
b_scaled = ... # multiply/expand b by b_scale

# Matrix multiply with the correct transpose
result = torch.matmul(a_scaled, b_scaled.T)

# Store as BF16 in c, ensuring row-major
c[...] = result.to(torch.bfloat16)
```

For details, check the provided code files and in particular, `reference.py` and your own `submission.py`.

---

## 🏃‍♂️ How to Run & Test

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/SaiDhanushKolla777/triton-fp8-gemm-kernel.git
   cd triton-fp8-gemm-kernel
   ```

2. **Edit `submission.py`:**
   Put your block-wise GEMM implementation or optimize further.

3. **Check Correctness:**
   ```python
   from reference import check
   check('submission.py')
   ```
   This will run the test harness and compare your outputs vs. the reference.

4. **(Optional) Benchmarking:**
   The `task.yml` describes the official benchmarks and their shapes. You can manually benchmark using the prescribed test cases and compare your execution time to the “speed of light” numbers.

---

## 🌟 Performance Tips

- **Start with correctness:** Use high-level code to make it correct first.
- **Optimize:** Once correct, use Triton or HIP/CUDA (see templates) for serious speed.
- **Test early and often:** Always check results against the reference before benchmarking.

---

## 📖 Additional Notes

- **CRLF/LF warnings** are normal when using Windows and can be ignored for this context.
- **Column-major vs. row-major:** Double-check data layout especially when moving between PyTorch (row-major by default) and the competition’s expected format.

---

## 🙏 Acknowledgements

- Thanks to AMD and the GPUmode organizers for providing the reference implementations, documentation, and a fun real-world challenge.
- Thanks to the developers of [PyTorch](https://pytorch.org/) and [Triton](https://github.com/openai/triton) for outstanding open-source tools.

---

## 📜 License

MIT License

---

**Questions or suggestions? Open an issue or pull request on this repo!**

---

This README should walk you or any collaborator through the challenge, the code structure, and how to proceed from start to finish.  
If you want code examples or walkthroughs for any file, feel free to ask!


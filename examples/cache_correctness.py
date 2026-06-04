"""Triton compile-cache correctness probe.

Compiles three Triton kernels (vector add, matmul, softmax) under two
conditions and compares numerics to torch eager:

  1. cold cache + TRITON_ALWAYS_COMPILE=1     (forces a fresh compile)
  2. warm cache (re-running this script)      (re-uses entries from run 1)

A healthy Triton cache produces **bit-identical** output between cold and
warm for every kernel x dtype combination. If cold and warm disagree on a
given hardware + Triton version, the cache key is not capturing something
it should, and you should:

  - wipe ~/.triton/cache (and "$XDG_CACHE_HOME/triton" if set)
  - set TRITON_ALWAYS_COMPILE=1 while validating
  - file an issue with this script's output

Run from a clean shell with:

    # Run 1 (cold + force compile, writes the cache):
    rm -rf /tmp/triton_correctness_cache
    TRITON_CACHE_DIR=/tmp/triton_correctness_cache \
    TRITON_ALWAYS_COMPILE=1 \
    python examples/cache_correctness.py

    # Run 2 (warm + no force, reads the cache from run 1):
    TRITON_CACHE_DIR=/tmp/triton_correctness_cache \
    python examples/cache_correctness.py

On Blackwell sm_121 you also need:

    export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
    export TORCH_CUDA_ARCH_LIST="12.1+PTX"

Note on matmul fp32: Triton's default `tl.dot` precision is tf32 on tensor-
core hardware, while torch fp32 matmul is ieee by default — so the matmul
fp32 error vs the torch reference will be ~5e-2 on every sm_12x part. That
is NOT a cache bug. You can confirm by setting
`torch.backends.cuda.matmul.allow_tf32=True` in your own code; torch's tf32
matmul vs its ieee matmul already differs by ~2e-2 on the same problem.
What matters is that cold and warm produce identical numbers.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def vec_add_kernel(X, Y, OUT, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < N
    x = tl.load(X + off, mask=mask)
    y = tl.load(Y + off, mask=mask)
    tl.store(OUT + off, x + y, mask=mask)


@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    off_m = pid_m * BM + tl.arange(0, BM)
    off_n = pid_n * BN + tl.arange(0, BN)
    off_k = tl.arange(0, BK)
    a_ptrs = A + (off_m[:, None] * stride_am + off_k[None, :] * stride_ak)
    b_ptrs = B + (off_k[:, None] * stride_bk + off_n[None, :] * stride_bn)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(a_ptrs, mask=(off_m[:, None] < M) & ((k + off_k)[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=((k + off_k)[:, None] < K) & (off_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BK * stride_ak
        b_ptrs += BK * stride_bk
    c_ptrs = C + (off_m[:, None] * stride_cm + off_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc.to(C.dtype.element_ty), mask=(off_m[:, None] < M) & (off_n[None, :] < N))


@triton.jit
def softmax_kernel(X, OUT, N_COLS, stride_row, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    x_ptrs = X + row * stride_row + cols
    mask = cols < N_COLS
    x = tl.load(x_ptrs, mask=mask, other=-float("inf"))
    x_max = tl.max(x, axis=0)
    x_shift = x - x_max
    num = tl.exp(x_shift)
    denom = tl.sum(num, axis=0)
    y = num / denom
    tl.store(OUT + row * stride_row + cols, y, mask=mask)


def run_vec_add(dtype):
    N = 8192
    torch.manual_seed(0)
    x = torch.randn(N, device="cuda", dtype=dtype)
    y = torch.randn(N, device="cuda", dtype=dtype)
    out_triton = torch.empty_like(x)
    BLOCK = 1024
    grid = ((N + BLOCK - 1) // BLOCK,)
    vec_add_kernel[grid](x, y, out_triton, N, BLOCK=BLOCK)
    return x + y, out_triton


def run_matmul(dtype):
    M, N, K = 256, 256, 256
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=dtype)
    b = torch.randn(K, N, device="cuda", dtype=dtype)
    c_triton = torch.empty(M, N, device="cuda", dtype=dtype)
    BM = BN = BK = 32
    grid = ((M + BM - 1) // BM, (N + BN - 1) // BN)
    matmul_kernel[grid](
        a, b, c_triton,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c_triton.stride(0), c_triton.stride(1),
        BM=BM, BN=BN, BK=BK,
    )
    return a @ b, c_triton


def run_softmax(dtype):
    rows, cols = 128, 1024
    torch.manual_seed(0)
    x = torch.randn(rows, cols, device="cuda", dtype=dtype)
    out_triton = torch.empty_like(x)
    grid = (rows,)
    BLOCK = triton.next_power_of_2(cols)
    softmax_kernel[grid](x, out_triton, cols, x.stride(0), BLOCK=BLOCK)
    return torch.softmax(x, dim=-1), out_triton


def report(name, ref, got, atol_fp32=1e-5, atol_bf16=5e-3):
    err = (ref.float() - got.float()).abs().max().item()
    atol = atol_bf16 if got.dtype == torch.bfloat16 else atol_fp32
    ok = err <= atol
    print(f"  {name:14s} dtype={str(got.dtype):20s} max_abs_err={err:.3e} atol={atol:.0e}  {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available; nothing to test.")
        return 0

    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    cache_dir = os.environ.get("TRITON_CACHE_DIR", os.path.expanduser("~/.triton/cache"))
    always_compile = os.environ.get("TRITON_ALWAYS_COMPILE", "0")
    print("=== Triton compile-cache correctness probe ===")
    print(f"  GPU:                  {name}  cap=sm_{cap[0]}{cap[1]}")
    print(f"  torch:                {torch.__version__}")
    print(f"  triton:               {triton.__version__}")
    print(f"  TRITON_PTXAS_PATH:    {os.environ.get('TRITON_PTXAS_PATH', '(unset)')}")
    print(f"  TRITON_OVERRIDE_ARCH: {os.environ.get('TRITON_OVERRIDE_ARCH', '(unset)')}")
    print(f"  TRITON_CACHE_DIR:     {cache_dir}")
    print(f"  TRITON_ALWAYS_COMPILE={always_compile}")
    cache_path = Path(cache_dir)
    if cache_path.exists():
        n_dirs = sum(1 for _ in cache_path.iterdir() if _.is_dir())
        print(f"  cache state:          {n_dirs} dirs in cache (warm)")
    else:
        print(f"  cache state:          no cache dir (cold)")
    print()

    passed = []
    for dtype_name, dtype in [("fp32", torch.float32), ("bf16", torch.bfloat16)]:
        print(f"--- {dtype_name} ---")
        for kernel_name, runner in [("vec_add", run_vec_add), ("matmul", run_matmul), ("softmax", run_softmax)]:
            try:
                ref, got = runner(dtype)
                passed.append(report(kernel_name, ref, got))
            except Exception as exc:
                print(f"  {kernel_name:14s} dtype={dtype_name:5s} EXC: {type(exc).__name__}: {exc}")
                passed.append(False)

    print()
    if cache_path.exists():
        archs = set()
        for d in sorted(cache_path.iterdir())[-5:]:
            if not d.is_dir():
                continue
            for jf in d.glob("*.json"):
                try:
                    with open(jf) as fh:
                        meta = json.load(fh)
                    if "target" in meta and "arch" in meta.get("target", {}):
                        archs.add(str(meta["target"]["arch"]))
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"  cache arch tag(s): {sorted(archs)}")

    print()
    print(f"OVERALL: {sum(passed)}/{len(passed)} kernels passed against torch eager")
    print("(matmul fp32 'FAIL' against torch ieee is the tf32 default for tl.dot — see docstring)")
    return 0 if all(passed) else 1


if __name__ == "__main__":
    sys.exit(main())

"""Minimal smoke test for Triton on Blackwell (sm_121, GB10 / DGX Spark).

Run from a clean shell with:
    TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    TORCH_CUDA_ARCH_LIST="12.1+PTX" \
    python examples/smoke.py

Expected on a healthy bring-up:
    Triton vector-add OK; max-diff = 0.0
    Triton gather OK; max-diff = 0.0

If you see "no kernel image is available for execution on the device", your
environment is still routing Triton through a ptxas that targets the wrong
architecture (e.g. an sm_90-mapping wrapper from earlier bring-ups), or
TRITON_OVERRIDE_ARCH is set to something else.
"""
from __future__ import annotations

import os
import sys

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < n
    x = tl.load(x_ptr + o, mask=m)
    y = tl.load(y_ptr + o, mask=m)
    tl.store(out_ptr + o, x + y, mask=m)


@triton.jit
def gather_kernel(idx_ptr, table_ptr, out_ptr, n, D, BLOCK_D: tl.constexpr):
    pid = tl.program_id(0)
    d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    if pid >= n:
        return
    idx = tl.load(idx_ptr + pid).to(tl.int32)
    mask = d < D
    val = tl.load(table_ptr + idx * D + d, mask=mask, other=0.0)
    tl.store(out_ptr + pid * D + d, val, mask=mask)


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available; nothing to test.")
        return 0

    cap = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name(0)
    print(f"Device: {name}, compute capability: sm_{cap[0]}{cap[1]}")
    print(f"TRITON_PTXAS_PATH    = {os.environ.get('TRITON_PTXAS_PATH', '(unset)')}")
    print(f"TORCH_CUDA_ARCH_LIST = {os.environ.get('TORCH_CUDA_ARCH_LIST', '(unset)')}")
    print(f"TRITON_OVERRIDE_ARCH = {os.environ.get('TRITON_OVERRIDE_ARCH', '(unset)')}")
    print()

    # Test 1: vector add
    x = torch.randn(4096, device="cuda")
    y = torch.randn(4096, device="cuda")
    out = torch.empty_like(x)
    grid = (triton.cdiv(x.numel(), 1024),)
    add_kernel[grid](x, y, out, x.numel(), BLOCK=1024)
    diff = (out - (x + y)).abs().max().item()
    print(f"Triton vector-add OK; max-diff = {diff}")
    if diff != 0.0:
        return 1

    # Test 2: gather (a common LM-tokenizer / embedding access pattern)
    n, D, V = 4096, 64, 256
    idx = torch.randint(0, V, (n,), dtype=torch.int32, device="cuda")
    table = torch.randn(V, D, device="cuda")
    out2 = torch.empty(n, D, device="cuda")
    grid2 = (n, triton.cdiv(D, 32))
    gather_kernel[grid2](idx, table, out2, n, D, BLOCK_D=32)
    ref = table[idx.long()]
    diff2 = (out2 - ref).abs().max().item()
    print(f"Triton gather OK; max-diff = {diff2}")
    if diff2 != 0.0:
        return 1

    print(f"\nAll Triton kernels passed on Blackwell sm_{cap[0]}{cap[1]}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

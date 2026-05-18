# triton-blackwell-bringup

Two environment variables make Triton compile and execute cleanly on
**NVIDIA Blackwell consumer / DGX-Spark / GB10 / GB200** parts (compute
capability `sm_121`) with PyTorch 2.9, even though PyTorch advertises its
maximum supported architecture as `sm_120`.

```bash
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas   # system CUDA 13 ptxas
export TORCH_CUDA_ARCH_LIST="12.1+PTX"               # declare arch + PTX fallback
# DO NOT set TRITON_OVERRIDE_ARCH — it forces wrong codegen
```

After exporting, Triton JIT compiles native `sm_121` kernels via the system
`ptxas` (shipped with CUDA 13.0+) and the resulting kernel images load and
run on the device.

> Verified on `NVIDIA GB10 (sm_121)` running PyTorch 2.9.0+cu130 / Triton 3.5
> / CUDA 13.0 (May 2026).

## Why this exists

PyTorch 2.9 (and a few months either side) caps its declared compute-capability
range at `sm_120`. On Blackwell parts that are `sm_121` you see:

```
torch/cuda/__init__.py: UserWarning: Found GPU0 NVIDIA GB10 which is of
cuda capability 12.1. Minimum and Maximum cuda capability supported by
this version of PyTorch is (8.0) - (12.0)
```

The runtime can still execute regular PyTorch ops fine. The pain point is
**Triton**, including any code path that lowers through it
(`torch.compile`, FlashAttention, hand-written kernels). Without the right
environment, Triton compiles to a stale target and you hit:

```
RuntimeError: Triton Error [CUDA]: no kernel image is available for
execution on the device
```

We tried four things that *do not* work:

| Attempt | What we did | Symptom |
|---|---|---|
| Do nothing | Just `import triton; @triton.jit` | `no kernel image …` |
| Override to sm_90 | `TRITON_OVERRIDE_ARCH=sm90` | `no kernel image …` (compiles for Hopper, can't load on Blackwell) |
| ptxas wrapper that rewrites `sm_121 → sm_90` | shell wrapper around ptxas | same as above |
| Strip `--embed-ptx` in the wrapper | (workaround in one forum) | random ptxas internal-compiler errors |

The actual root cause is that the older `ptxas` bundled with PyTorch's CUDA
toolkit doesn't know about `sm_121`. The system CUDA 13.0 toolkit's `ptxas`
**does** know about `sm_121` — telling Triton to use that one fixes it.

## Minimal smoke test

```bash
TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
TORCH_CUDA_ARCH_LIST="12.1+PTX" \
python examples/smoke.py
```

Expected output:

```
Device: NVIDIA GB10, compute capability: sm_121
Triton vector-add OK; max-diff = 0.0
Triton gather OK; max-diff = 0.0
All Triton kernels passed on Blackwell sm_121.
```

If you instead get `no kernel image is available for execution`, double-check
that `TRITON_OVERRIDE_ARCH` is unset and that `which ptxas` shows the same
binary as `TRITON_PTXAS_PATH`.

## Drop-in env block

Put this in your shell `rc` or a project `.env`:

```bash
# Triton bring-up for Blackwell sm_121 (GB10 / DGX Spark / consumer Blackwell)
# Requires CUDA 13.0+ system ptxas.
if [ -x /usr/local/cuda/bin/ptxas ]; then
    export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
fi
case ":$TORCH_CUDA_ARCH_LIST:" in
    *:12.1+PTX:*) ;;
    *) export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:+$TORCH_CUDA_ARCH_LIST;}12.1+PTX" ;;
esac
unset TRITON_OVERRIDE_ARCH
```

## Why this works (concise)

* PTX is forward-compatible. A binary emitted with `-arch=compute_121` plus the
  trailing `+PTX` directive can be re-JITted by the device driver, so the
  kernel runs on `sm_121` parts even when no precompiled SASS exists.
* The PyTorch-bundled `ptxas` was built before `sm_121` was added; the system
  CUDA 13.0 `ptxas` understands the target natively. Pointing Triton at the
  latter via `TRITON_PTXAS_PATH` keeps every other Triton internal unchanged.
* `TRITON_OVERRIDE_ARCH` forces Triton to *codegen* for a different ISA. Set
  to `sm_90`, kernels emit Hopper SASS that the Blackwell driver cannot load
  — hence the misleading "no kernel image" error.

## Doesn't help

* `torch.cuda.set_per_process_memory_fraction` — unrelated.
* `os.environ["CUDA_LAUNCH_BLOCKING"]=1` — masks but does not fix the issue.
* Downgrading to PyTorch 2.6 — Triton there is even older and fails harder.

## Citation

If you reference this workaround in published work, please cite the upstream
PyTorch issue/discussion linked in `REFERENCES.md`, and feel free to credit
this repo:

```
E. Niktab. Triton bring-up on Blackwell sm_121. 2026. github.com/eniktab/triton-blackwell-bringup
```

## License

MIT

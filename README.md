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

## Cache hygiene + cross-arch correctness

Per [@tbraun96 (Atlas)](https://github.com/triton-lang/triton/issues/10331#issuecomment-4615370311):
the Triton compile cache key does not always carry full compute capability,
so `~/.triton/cache` populated on one arch (or one toolchain) can be silently
reused on a different arch and miscompile. The failure mode is **worse than
"no kernel image"** — kernels load and compute wrong numbers.

### After any toolchain upgrade, container rebuild, or move between GPU classes

```bash
# Wipe every cache that might be keyed on the old arch / old toolchain
rm -rf ~/.triton/cache "$XDG_CACHE_HOME/triton" "$TMPDIR/triton_cache"
#   Also wipe any TRITON_CACHE_DIR you set explicitly on persistent storage.

# Force a fresh compile on every invocation while you validate.
# Drop this once you trust the cache key.
export TRITON_ALWAYS_COMPILE=1
```

### Empirical correctness probe

`examples/cache_correctness.py` compiles three Triton kernels
(`vec_add`, `matmul`, `softmax`) cold and warm and compares output to
`torch` eager. **Cold and warm must be bit-identical** before you trust the
cache.

Reference numbers from 2026-06-04:

|                 | cold sm_121 (Triton 3.5.0) | warm sm_121 | cold sm_90 (Triton 3.7.0) | warm sm_90 |
|-----------------|---------------------------:|------------:|--------------------------:|-----------:|
| vec_add fp32    | 0.00e+00                   | 0.00e+00    | 0.00e+00                  | 0.00e+00   |
| vec_add bf16    | 0.00e+00                   | 0.00e+00    | 0.00e+00                  | 0.00e+00   |
| softmax fp32    | 5.59e-9                    | 5.59e-9     | 5.59e-9                   | 5.59e-9    |
| softmax bf16    | 1.91e-6                    | 1.91e-6     | 1.91e-6                   | 1.91e-6    |
| matmul fp32     | 5.35e-2 †                  | 5.35e-2     | 6.33e-2 †                 | 6.33e-2    |
| matmul bf16     | 1.25e-1 †                  | 1.25e-1     | 0.00e+00                  | 0.00e+00   |

† tf32 default for `tl.dot`, not a cache or arch bug —
`torch.backends.cuda.matmul.allow_tf32=True` vs `False` alone differs by
2.1e-2 on the same matmul. Cold and warm are bit-identical on every
kernel × dtype × arch combination.

On disk, `<cache_dir>/<hash>/*.json` records `target.arch: 90` on H200 and
`target.arch: 121` on GB10, so the metadata at least knows the arch —
whether the lookup hash keys on arch would take more invasive instrumentation
to prove, but the on-disk evidence is consistent with the cache not silently
substituting across archs on Triton 3.5/3.7.

### Hard rule

**Never `TRITON_OVERRIDE_ARCH=<other-arch>` and never wrap `ptxas` to
silently downgrade `sm_12*` → `sm_90`.** Both produce kernels that load and
compute wrong rather than failing loud. If the bundled `ptxas` can't compile
for your real arch, point `TRITON_PTXAS_PATH` at a working `ptxas` —
don't lie about the arch.

## Doesn't help

* `torch.cuda.set_per_process_memory_fraction` — unrelated.
* `os.environ["CUDA_LAUNCH_BLOCKING"]=1` — masks but does not fix the issue.
* Downgrading to PyTorch 2.6 — Triton there is even older and fails harder.
* **Keeping `~/.triton/cache` around when changing arch or toolchain** —
  see "Cache hygiene + cross-arch correctness" above.

## Citation

If you reference this workaround in published work, please cite the upstream
PyTorch issue/discussion linked in `REFERENCES.md`, and feel free to credit
this repo:

```
E. Niktab. Triton bring-up on Blackwell sm_121. 2026. github.com/eniktab/triton-blackwell-bringup
```

## License

MIT

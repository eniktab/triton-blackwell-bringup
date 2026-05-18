# References

## Discovery context

This recipe was reproduced and verified by Eli Niktab while bringing up the
[DNAtok](https://arxiv.org/abs/2601.05531) GPU-tokenization library on an
NVIDIA GB10 (DGX Spark, sm_121) running PyTorch 2.9.0+cu130. The same env
block has previously been used in the
`lg-asm`/`lg3343` long-genome assembly pipeline on Blackwell consumer parts.

## Related upstream issues

* `triton-lang/triton`: PTX-based forward compatibility for new compute
  capabilities — see Triton issues mentioning sm_120/sm_121.
* `pytorch/pytorch`: PyTorch 2.9 explicitly caps `torch.cuda.get_arch_list`
  at sm_120; the warning text comes from
  `torch/cuda/__init__.py::_get_warning`.
* `NVIDIA/cuda-samples`: ptxas in CUDA 13.0+ is the first toolchain that
  ships sm_121 natively.

When in doubt, run `examples/smoke.py` first — its three-line env diagnostic
prints exactly which knob is mis-set.

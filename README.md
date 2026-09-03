# Triton for IBM POWER9 + NVIDIA Volta

**A working build of the [Triton](https://github.com/triton-lang/triton) GPU
compiler (v2.1.0) for `ppc64le` hosts driving sm_70 GPUs** — to our knowledge
the first Triton for the POWER architecture. It exists so that
Triton-dependent ML can run on POWER9+V100 systems (IBM AC922-class machines:
Bede, Summit-lineage clusters and similar), where no upstream wheels exist for
PyTorch, Triton, or mamba-ssm.

Everything below was verified on the [Bede](https://bede-documentation.readthedocs.io)
supercomputer (N8 CIR): POWER9 hosts, Tesla V100-SXM2-32GB, RHEL 8, Open-CE
PyTorch 2.1.2.

## What works (verified)

| Check | Result |
|---|---|
| `@triton.jit` kernel compiled on a POWER9 host, run on V100 (sm_70) | PASS |
| Mamba-2 SSD kernels (`mamba_kernels/`), forward **and** backward | max rel. err **2.1e-6** vs a pure-PyTorch reference |
| End-to-end: training a Mamba-2 byte-compression model on V100 | trained, deployed, losslessly verified |
| Whole-step CUDA-graph capture with these kernels inside the graph | PASS |

Measured against an RTX 5090 running identical fp32 workloads, the V100 lands
at a flat **2.2-2.5x** — the hardware generation gap, with no overhead
attributable to the port. On the V100 itself, the fused SSD scan is **21x
faster** than the pure-PyTorch fallback it replaces (6.8 ms vs 143.7 ms
fwd+bwd, 50k tokens).

## Install (prebuilt)

```bash
# python 3.11 + torch 2.1.x (on ppc64le that means Open-CE conda)
pip install <wheel from the Releases page>          # triton 2.1.0, ppc64le
pip install "git+https://github.com/boa-collaboration/triton-power9-volta@power9-volta"  # mamba_kernels
pip install einops packaging                        # mamba_kernels deps
```

No CUDA toolkit is needed at runtime — the wheel bundles the (ppc64le)
`ptxas`. Verify:

```bash
python ppc64le/hello_triton.py          # KERNEL RESULT: PASS
python ppc64le/test_mamba_kernels.py    # MAMBA KERNELS: PASS
```

## What changed vs upstream

**No compiler logic was modified.** The entire port is two commits of build
plumbing on top of the untouched v2.1.0 import:

1. `python/setup.py` skips the bundled-`ptxas` download (an x86_64 binary from
   the conda nvidia channel that cannot execute on POWER9); the build copies
   the CUDA toolkit's native `ptxas` instead.
2. `-Werror` is dropped (LLVM configure args and triton's own
   `CMAKE_CXX_FLAGS`): gcc 12 on ppc64le emits a false-positive
   `stringop-overflow` in MLIR's `Value.h` that aborts an otherwise clean
   build.

Diff it yourself: `git diff main power9-volta -- python/setup.py CMakeLists.txt`.

## `mamba_kernels/` — Mamba-2 SSD without mamba-ssm

`mamba-ssm` publishes no ppc64le wheels and its package drags compiled CUDA
extensions. But a Mamba-2 SSD trainer needs exactly two Triton entry points,
so they are vendored here from mamba-ssm 2.2.5 (Apache-2.0, Tri Dao & Albert
Gu — see `mamba_kernels/LICENSE`), byte-identical except for import rewrites,
and verified `torch.equal` against the original package on x86:

```python
from mamba_kernels import mamba_chunk_scan_combined, RMSNormGated
```

## Building from source

Full recipe with the exact Slurm job: [`PPC64LE.md`](PPC64LE.md). The
essentials:

* LLVM at triton's pinned commit `c5dede880d175f7229c9b2923f4753e12702305d`,
  `-DLLVM_ENABLE_PROJECTS=mlir`,
  `-DLLVM_TARGETS_TO_BUILD="PowerPC;NVPTX;AMDGPU"` — **AMDGPU is mandatory**
  (triton's HSACO layer links its symbols unconditionally). ~20 min on 8
  POWER9 cores.
* Build triton as a **wheel** (`pip wheel . --no-build-isolation`); editable
  installs misplace `triton._C`.
* Copy `$CUDA_HOME/bin/ptxas` into `python/triton/third_party/cuda/bin/`
  before building, so the wheel bundles the native one.
* Never run tests with the source tree on `sys.path`/cwd — the `triton/`
  directory shadows the installed package.

## Known limitations

* **`torch.compile` (inductor) does not work** with this wheel on torch 2.1:
  dynamo introspects triton-internal APIs and expects the `pytorch-triton`
  fork, not the upstream tag. Direct `@triton.jit` kernels — including
  everything in `mamba_kernels/` — are unaffected, and whole-step CUDA graphs
  work. A wheel built from pytorch's pinned triton commit should close this.
* First use of a large autotuned kernel family (e.g. the SSD scan) spends
  minutes in autotuning; results are cached under `~/.triton`.
* The published wheel targets cp311 + torch 2.1.x; validation was on sm_70.
  Other compute capabilities should work (NVPTX + AMDGPU are built in) but
  are untested here. Upstream's full test suite has not been run on ppc64le;
  validation is the conformance tests above plus a real training workload.

## Provenance

* Triton is (c) the Triton developers, MIT license (`LICENSE`); this branch
  carries the unmodified v2.1.0 import in its history (`main`).
* `mamba_kernels/` is Apache-2.0, (c) 2024 Tri Dao, Albert Gu.
* Built and verified on Bede, the N8 CIR supercomputer (EPSRC EP/T022167/1).
* Upstream's original README: [`README.upstream.md`](README.upstream.md).

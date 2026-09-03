# Triton 2.1.0 for POWER9 + Volta (ppc64le, sm_70)

This branch makes Triton 2.1.0 build and run on IBM POWER9 hosts driving
NVIDIA V100s — verified on the Bede (N8 CIR) cluster: `@triton.jit` kernels
compile and run on sm_70, and the vendored Mamba-2 SSD kernels
(`mamba_kernels/`) pass conformance to 2e-6 against a pure-PyTorch reference,
forward and backward. A `bitmamba2` model was trained end-to-end with them.

No compiler logic was changed. The whole port is:

1. **`python/setup.py`**: the bundled `ptxas` download is an x86_64 binary from
   the conda nvidia channel; it is skipped, and the CUDA toolkit's native
   `ptxas` is copied in at build time instead.
2. **No `-Werror`** (setup.py LLVM args + top-level `CMakeLists.txt`): gcc 12
   on ppc64le emits a false-positive `stringop-overflow` in MLIR's `Value.h`.
3. Build recipe facts below (the pinned LLVM must include **AMDGPU** — triton
   links `LLVMInitializeAMDGPUTarget` unconditionally via its HSACO layer).

## Installing the prebuilt wheel (users)

```bash
pip install triton-2.1.0-cp311-cp311-linux_ppc64le.whl   # from this repo's Releases
python -c "import triton; print(triton.__version__)"
```

Requires: python 3.11, torch 2.1.x (Open-CE on ppc64le), a CUDA toolkit on
PATH at runtime is NOT needed (ptxas is bundled — the ppc64le one).
The Mamba kernels additionally need `pip install einops packaging`.

## Building from source (maintainers)

LLVM first — triton 2.1.0 pins commit `c5dede880d175f7229c9b2923f4753e12702305d`:

```bash
git init llvm-project && cd llvm-project
git remote add origin https://github.com/llvm/llvm-project.git
git fetch --depth 1 origin c5dede880d175f7229c9b2923f4753e12702305d
git checkout FETCH_HEAD
cmake -G Ninja -S llvm -B build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$LLVM_PREFIX \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DLLVM_TARGETS_TO_BUILD="PowerPC;NVPTX;AMDGPU" \
  -DLLVM_ENABLE_ASSERTIONS=OFF -DLLVM_ENABLE_TERMINFO=OFF -DLLVM_ENABLE_ZSTD=OFF \
  -DLLVM_INSTALL_UTILS=ON -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF \
  -DLLVM_INCLUDE_BENCHMARKS=OFF -DMLIR_INCLUDE_TESTS=OFF \
  -DLLVM_PARALLEL_LINK_JOBS=4
ninja -C build && ninja -C build install    # ~20 min on 8 POWER9 cores
```

Then triton, as a wheel (editable installs misplace `triton._C` — always wheel):

```bash
cd python
mkdir -p triton/third_party/cuda/bin
cp $CUDA_HOME/bin/ptxas triton/third_party/cuda/bin/ptxas   # the ppc64le one
export LLVM_SYSPATH=$LLVM_PREFIX MAX_JOBS=32
pip wheel . --no-build-isolation --no-deps -w dist/
pip install dist/triton-*.whl
```

`ppc64le/build_bede.sbatch` is the exact Slurm job used on Bede (resumable,
flock-guarded). Gotchas that cost real debugging time: never run tests with the
source tree on `sys.path`/cwd (the `triton/` dir shadows the installed
package); use a per-job `TORCH_EXTENSIONS_DIR`; expect the FIRST training run
to spend several minutes in triton autotune JIT (cached afterwards).

## Verifying

```bash
python ppc64le/hello_triton.py          # KERNEL RESULT: PASS
python ppc64le/test_mamba_kernels.py    # MAMBA KERNELS: PASS  (rel err ~2e-6 on V100)
```

## `mamba_kernels/` — Mamba-2 SSD without mamba-ssm

`mamba_ssm` cannot be installed on ppc64le (its package pulls compiled CUDA
extensions and more). But a Mamba-2 SSD trainer needs exactly two entry points,
both pure Triton: `mamba_chunk_scan_combined` and the gated `RMSNorm`. They are
vendored here (from mamba-ssm 2.2.5, imports rewritten, otherwise byte-for-byte
— outputs verified `torch.equal` against the original package on x86):

```python
from mamba_kernels import mamba_chunk_scan_combined, RMSNormGated
```

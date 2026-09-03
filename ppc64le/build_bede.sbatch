#!/bin/bash
#SBATCH --account=bdman38
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --job-name=triton_ppc
# Build LLVM 17 (triton 2.1.0's pin) + triton on ppc64le, then run a
# hello-world kernel on the V100. Resumable: each stage skips if already done.
set -e
# serialize concurrent build jobs (infer + gpu partitions share this dir):
# whoever gets the lock builds; the other waits, then finds the stages done
# and skips straight to the kernel test on ITS OWN GPU type.
exec 9>/nobackup/projects/bdman38/tritonbuild/.lock
flock 9
module load gcc/12.2
module load cuda/12.4.1
source /nobackup/projects/bdman38/miniconda3/etc/profile.d/conda.sh
conda activate boa-ppc

B=/nobackup/projects/bdman38/tritonbuild
PREFIX=$B/llvm-install
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
echo "gcc: $(gcc --version|head -1) | cmake: $(cmake --version|head -1) | nvcc: $(nvcc --version|tail -2|head -1)"
echo "cores: $(nproc)"

############################ 1. LLVM 17 + MLIR ############################
if [ -f "$PREFIX/bin/llvm-config" ]; then
  echo "=== LLVM already built at $PREFIX — skipping ==="
else
  echo "=== building LLVM 17 + MLIR (this is the long pole) ==="
  mkdir -p $B/llvm-build && cd $B/llvm-build
  cmake -G Ninja $B/llvm-project/llvm \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$PREFIX \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_TARGETS_TO_BUILD="PowerPC;NVPTX;AMDGPU" \
    -DLLVM_ENABLE_ASSERTIONS=OFF \
    -DLLVM_ENABLE_TERMINFO=OFF \
    -DLLVM_ENABLE_ZSTD=OFF \
    -DLLVM_ENABLE_ZLIB=ON \
    -DLLVM_INSTALL_UTILS=ON \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DLLVM_INCLUDE_EXAMPLES=OFF \
    -DLLVM_INCLUDE_BENCHMARKS=OFF \
    -DMLIR_INCLUDE_TESTS=OFF \
    -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
    -DLLVM_PARALLEL_LINK_JOBS=4
  time ninja -j$(nproc)
  ninja install
  echo "=== LLVM installed: $($PREFIX/bin/llvm-config --version) ==="
fi

############################ 2. triton 2.1.0 ############################
cd $B/triton/python

# (a) the bundled ptxas is downloaded as an x86_64 binary and cannot run here;
#     pre-place the ppc64le one from the CUDA module and neuter the download.
mkdir -p triton/third_party/cuda/bin triton/third_party/cuda/include
cp -f $CUDA_HOME/bin/ptxas triton/third_party/cuda/bin/ptxas
chmod +x triton/third_party/cuda/bin/ptxas
for h in cuda.h; do
  [ -f triton/third_party/cuda/include/$h ] || cp -f $CUDA_HOME/include/$h triton/third_party/cuda/include/ 2>/dev/null || true
done
python - <<'PY'
import re, pathlib
p = pathlib.Path("setup.py"); s = p.read_text()
if "def download_and_copy_ptxas():\n    return  # ppc64le" not in s:
    s = s.replace("def download_and_copy_ptxas():\n",
                  "def download_and_copy_ptxas():\n    return  # ppc64le: use the system ptxas placed by the build script\n", 1)
    p.write_text(s); print("  patched setup.py: ptxas download disabled")
else:
    print("  setup.py already patched")
PY

export LLVM_SYSPATH=$PREFIX
export TRITON_BUILD_WITH_CLANG_LLD=0
export MAX_JOBS=32                       # setup.py otherwise uses -j256
# gcc-12/ppc64le emits a false-positive stringop-overflow in MLIR Value.h;
# setup.py is patched to LLVM_ENABLE_WERROR=OFF, and this covers any stragglers
export CXXFLAGS="-Wno-error=stringop-overflow"
echo "=== building the triton WHEEL (the shippable artifact) ==="
mkdir -p $B/dist
time pip wheel . --no-build-isolation --no-deps -w $B/dist
pip install --force-reinstall --no-deps $B/dist/triton-*.whl
ls -l $B/dist/

############################ 3. the decisive tests ############################
# @triton.jit needs inspect.getsource, so the kernel MUST live in a real file
cat > $B/hello_triton.py <<'HELLO'
import torch, triton, triton.language as tl
print("triton", triton.__version__, "| torch", torch.__version__,
      "|", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=m) + tl.load(y_ptr + offs, mask=m), mask=m)

n = 4096
x = torch.rand(n, device="cuda"); y = torch.rand(n, device="cuda")
out = torch.empty_like(x)
add_kernel[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)
torch.cuda.synchronize()
ok = torch.allclose(out, x + y)
print("KERNEL RESULT:", "PASS - triton compiles and runs on this GPU" if ok else "FAIL - wrong output")
HELLO
mkdir -p $B/testrun && cd $B/testrun     # no triton/ dir here -> no shadowing
echo "=== hello-world on $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
python $B/hello_triton.py

echo "=== phase 3: vendored mamba SSD kernels on this GPU ==="
python $B/test_mamba_kernels.py

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
assert torch.allclose(out, x + y), "wrong output"
print("KERNEL RESULT: PASS")

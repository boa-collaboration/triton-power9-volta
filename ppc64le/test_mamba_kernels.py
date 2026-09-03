"""Conformance: the vendored SSD kernels vs a pure-torch reference, on this GPU.
Run from the repo root (or anywhere NOT containing a triton/ source dir)."""
import sys, pathlib, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # repo root -> mamba_kernels
from mamba_kernels import mamba_chunk_scan_combined, RMSNormGated


def ssd_ref(x, dt, A, B, C, chunk, D):
    """minimal chunked SSD reference (fp32), matches mamba_chunk_scan_combined."""
    b, L, H, P = x.shape
    N = B.shape[-1]
    B = B[:, :, 0].float(); C = C[:, :, 0].float()
    x = x.float(); dt = dt.float(); A = A.float()
    y = torch.zeros(b, L, H, P, device=x.device)
    h = torch.zeros(b, H, N, P, device=x.device)
    for t0 in range(0, L, chunk):
        t1 = min(t0 + chunk, L)
        for t in range(t0, t1):
            dA = (dt[:, t] * A).exp()                                  # [b,h]
            h = h * dA[:, :, None, None] + torch.einsum(
                "bn,bhp->bhnp", B[:, t], x[:, t] * dt[:, t][:, :, None])
            y[:, t] = torch.einsum("bn,bhnp->bhp", C[:, t], h)
    if D is not None:
        y = y + x.float() * D[None, None, :, None]
    return y


b, L, H, P, N, CH = 2, 512, 4, 64, 16, 128
torch.manual_seed(0)
x = torch.randn(b, L, H, P, device="cuda", requires_grad=True)
dt = torch.rand(b, L, H, device="cuda").add_(0.01)
A = -torch.rand(H, device="cuda") - 0.1
B = torch.randn(b, L, 1, N, device="cuda"); C = torch.randn(b, L, 1, N, device="cuda")
D = torch.randn(H, device="cuda")

y = mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size=CH, D=D, z=None,
                              dt_bias=None, dt_softplus=False)
y.sum().backward()
yr = ssd_ref(x.detach(), dt, A, B, C, CH, D)
rel = ((y - yr).abs().max() / yr.abs().max()).item()
print(f"{torch.cuda.get_device_name(0)} {torch.cuda.get_device_capability(0)}")
print(f"scan fwd+bwd max rel err vs reference: {rel:.2e}")
n = RMSNormGated(P, eps=1e-5, norm_before_gate=False).cuda()
n(y.detach(), torch.randn_like(y)).sum()
print("MAMBA KERNELS:", "PASS" if rel < 1e-3 else "FAIL")

"""mamba-ssm 2.2.5 SSD Triton kernels, vendored standalone — no mamba_ssm
package required. Exactly the two entry points bitmamba2 uses."""
from .ssd_combined import mamba_chunk_scan_combined
from .layernorm_gated import RMSNorm as RMSNormGated

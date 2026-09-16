"""VMamba VSSBlock adapter for HFSR-Net.

The upstream VMamba source is vendored at ``third_party/VMamba``.  This adapter
keeps Ultralytics tensors in NCHW format and delays importing VMamba until a
model is constructed, so that the project source remains inspectable even when
the optional CUDA extension is not installed.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import torch.nn as nn

from .conv import Conv

__all__ = ("VSSBlock",)


class _OutOfPlaceSiLU(nn.SiLU):
    """SiLU variant Ultralytics will not globally switch to unsafe inplace mode."""

    def __init__(self):
        super().__init__(inplace=False)


def _load_vmamba_module():
    """Load the vendored upstream VMamba single-file implementation once."""
    module_name = "hfsr_vmamba_upstream"
    if module_name in sys.modules:
        return sys.modules[module_name]
    source = Path(__file__).resolve().parents[3] / "third_party" / "VMamba" / "vmamba.py"
    if not source.is_file():
        raise FileNotFoundError(f"Vendored VMamba source was not found: {source}")
    spec = spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load vendored VMamba source: {source}")
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        sys.modules.pop(module_name, None)
        raise ImportError(
            "VMamba dependencies are unavailable. See third_party/VMamba/requirements.txt "
            "and build third_party/VMamba/kernels/selective_scan when preparing the runtime environment."
        ) from error
    return module


class VSSBlock(nn.Module):
    """NCHW adapter around VMamba's upstream Visual State Space block."""

    def __init__(
        self,
        c1,
        c2,
        d_state=16,
        ssm_ratio=2.0,
        d_conv=3,
        mlp_ratio=4.0,
        drop_path=0.0,
        forward_type="v3",
    ):
        super().__init__()
        # ``v3`` uses the vendored selective_scan_cuda_oflex extension built by
        # third_party/VMamba/kernels/selective_scan/setup.py.  The upstream
        # ``v2`` alias expects a different "core" extension which is not built
        # by this repository's default setup.
        vmamba = _load_vmamba_module()
        self.proj = Conv(c1, c2, 1) if c1 != c2 else nn.Identity()
        self.block = vmamba.VSSBlock(
            hidden_dim=c2,
            channel_first=True,
            ssm_d_state=d_state,
            ssm_ratio=ssm_ratio,
            ssm_conv=d_conv,
            ssm_act_layer=_OutOfPlaceSiLU,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            forward_type=forward_type,
        )

    def forward(self, x):
        """Apply the VSS block while preserving Ultralytics' NCHW convention."""
        return self.block(self.proj(x))

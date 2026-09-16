#!/usr/bin/env python3
"""Build HFSR-Net and run a small forward/backward environment smoke test."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import platform
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != ROOT]
sys.path.insert(0, str(ROOT))
DEFAULT_MODEL = ROOT / "ultralytics/cfg/models/v8/yolov8s-hfsrnet.yaml"


def tensor_sum(value):
    """Return a scalar sum for nested model outputs."""
    if torch.is_tensor(value):
        return value.float().sum()
    if isinstance(value, dict):
        return sum((tensor_sum(item) for item in value.values()), torch.tensor(0.0))
    if isinstance(value, (list, tuple)):
        return sum((tensor_sum(item) for item in value), torch.tensor(0.0))
    raise TypeError(f"Unsupported model output type: {type(value)!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--imgsz", type=int, default=64)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--require-cuda-extension", action="store_true")
    args = parser.parse_args()

    extension_available = importlib.util.find_spec("selective_scan_cuda_oflex") is not None
    print(f"platform={platform.platform()}")
    print(f"python={platform.python_version()} torch={torch.__version__} cuda={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()} oflex_extension={extension_available}")
    if args.require_cuda_extension and not extension_available:
        raise SystemExit("selective_scan_cuda_oflex is not importable; build the vendored extension first")

    from ultralytics import YOLO

    wrapper = YOLO(str(args.model))
    model = wrapper.model.to(args.device)
    model.train(args.backward)
    x = torch.randn(1, 3, args.imgsz, args.imgsz, device=args.device)
    output = model(x)
    if args.backward:
        tensor_sum(output).backward()
    print(f"HFSR_SMOKE_OK device={args.device} imgsz={args.imgsz} backward={args.backward}")


if __name__ == "__main__":
    main()

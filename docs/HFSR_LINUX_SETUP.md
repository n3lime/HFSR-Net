# HFSR-Net Linux setup

HFSR-Net uses the vendored VMamba `VSSBlock`. The supported accelerated path
in this repository is the vendored `selective_scan_cuda_oflex` CUDA extension.
When that extension is unavailable, the model falls back to a much slower pure
PyTorch selective scan so that model structure can still be tested.

## Recommended environment

The reproducible baseline is Python 3.10, PyTorch 2.2.0, torchvision 0.17.0,
and CUDA 12.1. A C++ compiler and a CUDA 12.x development toolkit containing
`nvcc` are also required; the conda CUDA runtime alone is not sufficient to
compile the extension.

```bash
conda env create -f environment-hfsr-linux.yml
conda activate hfsr
bash scripts/setup_hfsr_linux.sh
```

The setup script installs this repository in editable mode, compiles the
vendored extension, and runs a small forward/backward smoke test. It stops with
an error if CUDA, `nvcc`, or the compiled extension is unavailable.

To rerun only the validation:

```bash
python scripts/check_hfsr.py \
  --device cuda --imgsz 64 --backward --require-cuda-extension
```

## Dataset path

Windows drive paths do not work on Linux. Copy the dataset YAML and replace its
`path` value with the absolute Linux dataset directory, for example:

```yaml
path: /data/Small Target PWD Detection Dataset
train: images/train
val: images/val
test: images/test
names:
  0: unhealthy
```

Then run a one-epoch smoke train before starting a long experiment:

```bash
yolo detect train \
  model=ultralytics/cfg/models/v8/yolov8s-hfsrnet.yaml \
  data=/path/to/dataset-linux.yaml \
  epochs=1 fraction=0.05 imgsz=128 batch=2 workers=2 device=0 amp=False
```

After that succeeds, use the intended image size, batch size, epoch count, and
AMP setting. Pure PyTorch fallback is for diagnostics only and is not
recommended for full training.

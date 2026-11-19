# NavAR

Floor segmentation for AR-style indoor/outdoor navigation (APS360 course project).

A U-Net-style CNN with an ImageNet-pretrained ResNet-34 encoder, trained on ADE20K
with labels merged into **floor / not floor / ignore**, and compared against a simple
colour-based baseline.

## Layout

```
src/        model, dataset and training code
scripts/    command-line utilities (data prep, environment checks)
configs/    training configs
notebooks/  exploration and figures
data/       datasets and cached masks (not tracked)
```

## Setup

Requires an NVIDIA GPU with a recent driver. The PyTorch wheels bundle CUDA, so no
system CUDA toolkit is needed.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision
pip install -r requirements.txt
python scripts/check_gpu.py
```

## Data

```bash
python scripts/prepare_ade20k.py
```

Downloads ADE20K SceneParsing (ADEChallengeData2016, ~1 GB) into `data/raw/`, then
writes merged masks to `data/ade20k_floor/masks/{training,validation}/` with
0 = not floor, 1 = floor, 255 = ignore. Which ADE20K classes count as floor or
ignore is set by name in `configs/labels.yaml`; their indices are looked up in the
dataset's `objectInfo150.txt`. `data/ade20k_floor/meta.json` records the resolved
mapping and pixel counts per split. Re-running skips masks that already exist;
pass `--overwrite` after changing the label config.

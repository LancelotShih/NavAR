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

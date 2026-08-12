# CARL (Sony インターン用フォーク)

> **目的:** ZenSa（自社モデル）と CARL の比較実験を行うためのフォークです。  
> オリジナル: https://github.com/IMSY-DKFZ/CARL

---

## 🚀 クイックセットアップ (Windows)

```powershell
# 1. クローン
git clone https://github.com/Ying1Lu/CARL.git
cd CARL

# 2. 仮想環境作成
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. PyTorch (CUDA 12.1) インストール
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. 依存パッケージ
pip install -r requirements.txt
pip install geobench --no-deps

# 5. SSL事前学習チェックポイントのダウンロード (1.33 GB)
#    ※ urllib/python-wgetは途中で切断されることがあるため curl 推奨
curl.exe -L -C - --retry 10 --retry-all-errors --retry-delay 5 -o carl_ssl_checkpoint.ckpt "https://zenodo.org/records/18671944/files/ssl_checkpoint_carl.ckpt"
```

### 動作確認テスト

```powershell
# チェックポイント不要 — モデル構築 + 7/33band推論 + GPU確認
python test_carl_basic.py

# SSL事前学習済み重みのロード + 推論（要 carl_ssl_checkpoint.ckpt）
python test_carl_ssl.py
```

### 確認済み環境
| 項目 | バージョン |
|------|-----------|
| Python | 3.10 |
| PyTorch | 2.5.1+cu121 |
| timm | 1.0.28 |
| pytorch-lightning | 2.6.5 |
| GPU | RTX 4070 Ti SUPER (327 MB / 453 MB peak) |

---

## 🎯 HSIRS (33band) でセグメンテーションを行う手順

### 概要

```
┌─────────────────────────────────────────────────────────┐
│  SSL事前学習済みエンコーダ (carl_ssl_checkpoint.ckpt)     │
│  → Spectral Encoder + Spatial Encoder  [frozen]         │
└───────────────────────────┬─────────────────────────────┘
                            │ 特徴マップ [B, 768, H/8, W/8]
                            ▼
┌─────────────────────────────────────────────────────────┐
│  セグメンテーションヘッド (Conv2d 1×1)  [trainable]      │
│  → n_classes チャンネルに射影                             │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
                  セグメンテーションマスク出力
```

**重要:** SSL チェックポイントはエンコーダのみ。セグメンテーションには **GTマスク付きデータで Fine-tuning（Linear Probing）** が必要。

### Step 1: データセットクラスの作成

`carl/data/HSIRS.py` を作成する:

```python
"""HSIRS 33-band hyperspectral segmentation dataset."""
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset

NORMALIZATION_EPSILON = 1e-6

class HSIRS(Dataset):
    """HSIRS 33-band dataset for semantic segmentation.
    
    期待するディレクトリ構造:
        root_dir/
        ├── images/       # .npy files, shape (33, H, W)
        ├── labels/       # .npy files, shape (H, W), class IDs [0, n_classes-1]
        └── wavelengths.npy  # shape (33,), 波長 [μm] 単位
    """
    
    def __init__(self, root_dir: str, split: str = 'train', cfg=None):
        self.root_dir = Path(root_dir) / split
        self.cfg = cfg
        self.n_classes = cfg['model_kwargs']['n_classes']
        
        self.image_files = sorted((self.root_dir / 'images').glob('*.npy'))
        self.label_files = sorted((self.root_dir / 'labels').glob('*.npy'))
        
        # 波長情報 (μm単位)
        wl_path = Path(root_dir) / 'wavelengths.npy'
        self.wavelengths = np.load(wl_path).astype(np.float32)
        
        assert len(self.image_files) == len(self.label_files), \
            f"画像数 {len(self.image_files)} != ラベル数 {len(self.label_files)}"
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img = np.load(self.image_files[idx]).astype(np.float32)  # (33, H, W)
        label = np.load(self.label_files[idx]).astype(np.int64)  # (H, W)
        
        # Per-image normalization
        mean, std = img.mean(), img.std()
        img = (img - mean) / (std + NORMALIZATION_EPSILON)
        
        img_tensor = torch.from_numpy(img)
        wl_tensor = torch.from_numpy(self.wavelengths.copy())
        label_tensor = torch.from_numpy(label)
        
        return img_tensor, wl_tensor, label_tensor
```

### Step 2: 設定ファイルの作成

`configs/config_seg_hsirs.yaml`:

```yaml
model_kwargs:
  patch_size: 8
  image_size: 128          # 画像を128×128にリサイズ or クロップ
  n_classes: 5             # ← HSIRSのクラス数に合わせて変更
  spec_encoder_kwargs:
    embed_dim: 384
    depth: 8
    num_heads: 6
    layer_scale: 0.0001
    pos_enc_sigma: 3
    n_queries: 8
    qkv_bias: true
    ffn_bias: true
    drop_path_rate: 0.
    proj_drop: 0.
    drop: 0.
    attn_drop: 0.
  spat_encoder_kwargs:
    model_name: timm/eva02_base_patch14_224.mim_in22k
    depth: 8
    model_kwargs:
      drop_rate: 0.
      pos_drop_rate: 0.
      patch_drop_rate: 0.
      proj_drop_rate: 0.
      attn_drop_rate: 0.
      drop_path_rate: 0.

data_kwargs:
  train_dataset:
    name: HSIRS
    root_dir: /path/to/hsirs_dataset    # ← 実際のパスに変更
    split: train
  val_dataset:
    name: HSIRS
    root_dir: /path/to/hsirs_dataset
    split: val
  test_dataset:
    name: HSIRS
    root_dir: /path/to/hsirs_dataset
    split: test

training_kwargs:
  batch_size: 16
  num_workers: 4
  ssl_ckpt_path: carl_ssl_checkpoint.ckpt
  monitor_metric: val_mIoU
  log_dir: logs/hsirs_seg
  learning_rate: 1e-3

lightning_kwargs:
  max_epochs: 50
  accelerator: gpu
  devices: 1
  precision: bf16-mixed
  check_val_every_n_epoch: 5
  log_every_n_steps: 10
  enable_checkpointing: true
  num_sanity_val_steps: 0
  enable_progress_bar: true
```

### Step 3: データの準備

HSIRSデータを以下のフォルダ構造に変換:

```
hsirs_dataset/
├── wavelengths.npy          # shape (33,), μm単位 例: [0.40, 0.42, ..., 1.00]
├── train/
│   ├── images/              # {scene_id}.npy  shape (33, 128, 128)
│   └── labels/              # {scene_id}.npy  shape (128, 128)
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

> **注意:** 波長は必ず **マイクロメートル (μm)** 単位で指定すること。  
> 例: 400nm → 0.40μm, 1000nm → 1.00μm

### Step 4: 学習の実行

```powershell
python main_seg.py --config configs/config_seg_hsirs.yaml
```

### Step 5: 推論・GT比較

学習後、`logs/hsirs_seg/` に best checkpoint が保存される。  
推論スクリプト例:

```python
import torch
import numpy as np
from carl.trainer.seg_trainer import LinearTrainer
from carl.config import load_config

config = load_config("configs/config_seg_hsirs.yaml")
model = LinearTrainer.load_from_checkpoint("logs/hsirs_seg/.../best.ckpt", config=config)
model.eval()
model.cuda()

# 推論
img = torch.from_numpy(np.load("test_image.npy")).unsqueeze(0).cuda()   # (1,33,128,128)
wl  = torch.from_numpy(np.load("wavelengths.npy")).unsqueeze(0).cuda()  # (1,33)

with torch.no_grad():
    spatial_feat, _ = model.model(img, wl)
    pred = model.classifier(spatial_feat)             # (1, n_classes, 16, 16)
    pred_mask = pred.argmax(dim=1)                     # (1, 16, 16)
    # Upsample to original size
    pred_mask = torch.nn.functional.interpolate(
        pred_mask.unsqueeze(1).float(), size=(128,128), mode='nearest'
    ).squeeze().long()
```

---

[![arXiv](https://img.shields.io/badge/arXiv-2504.19223-b31b1b.svg)](https://arxiv.org/abs/2504.19223)
[![Conference](https://img.shields.io/badge/ICLR-2026-blue)](https://iclr.cc/virtual/2026/poster/10009281)

### Camera-Agnostic Representation Learning for Spectral Image Analysis

✔ Sensor-agnostic   
✔ Self-supervised pretraining  
✔ Classification  
✔ Segmentation  
✔ Satellite imaging SSL-checkpoint  

### 📄 Publication:
Accepted at **ICLR 2026**  

**Authors:** Alexander Baumann, Leonardo Ayala, Silvia Seidlitz, Jan Sellner,
Alexander Studier-Fischer, Berkin Özdemir, Lena Maier-Hein*, Slobodan Ilic*

**Paper:** https://arxiv.org/abs/2504.19223  

![CARL](readme_images/carl.png)

---

## Overview

CARL is a camera-agnostic feature encoder for spectral images. It supports downstream tasks such as **classification**, **segmentation**, and **regression**, and provides a **self-supervised checkpoint** for rapid transfer learning.

This repository contains model code, dataset loaders, and training implementations using **PyTorch Lightning**.

## Contents
- [Installation](#installation)
- [Quick start](#quick-start)
  - [Supervised training (classification / segmentation)](#supervised-training-classification--segmentation)
  - [Self-supervised training](#self-supervised-training)
- [Data format](#data-format)
- [Project structure (high level)](#project-structure-high-level)
- [License and third-party code](#license-and-third-party-code)
- [Citation](#citation)

## Installation

Optional: create an isolated virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install pytorch with CUDA support (adjust the CUDA version in the URL as needed), then install the required packages:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cuxxx
pip install -r requirements.txt
pip install geobench --no-deps
```

Optional: compile deformable attention CUDA kernels (needed for ViT-Adapter):

```bash
cd segmentation_heads/upernet/utils/ops
python setup.py build install
```

## Quick start

### Supervised training (classification / segmentation)

The repository supports supervised training for image classification and segmentation using CARL.

- **Spectral encoder (CARL):**
  - either randomly initialize, or
  - load the self-supervised CARL checkpoint (pretrained on remote sensing data)

- **Spatial encoder:**
  Choose among pretrained ViT/EVA models (via `timm`). Examples include **DINOv2, DINOv3, Perception Encoder, and EVA-02**.

- **Segmentation head (for semantic segmentation):**
  - Linear head
  - ViT-Adapter + UperNet
  - ViT-Adapter + Mask2Former
  - Swin Transformer + Mask2Former

1) Download the remote sensing self-supervised checkpoint (optional):

```bash
wget -O carl_ssl_checkpoint.ckpt https://zenodo.org/records/18671944/files/ssl_checkpoint_carl.ckpt 
```

2) Pick an example configuration in `configs/` or create your own.

3) Train:

Segmentation:
```bash
python main_seg.py --config configs/config_seg.yaml
```

Classification:
```bash
python main_cls.py --config configs/config_cls.yaml
```

Tip: for a minimal model load + feature extraction example, see `example.py`.

### Self-supervised training

To create your own self-supervised checkpoint on your custom data and model, prepare your datasets and configuration, then run:

```bash
python main_ssl.py --config configs/config_ssl.yaml
```

## Model performance
The performance of CARL-SSL was evaluated via linear probing across 11 diverse datasets and compared against 6 state-of-the-art baselines. As shown in the table below, CARL-SSL demonstrates robust generalization capabilities across different sensors, achieving an average rank of 1.6.

|Dataset|m-ben|m-eurosat|m-forestnet|m-crop-type|SegMunich|Wuhan|LoveDA Rural|WHU-OHS|Avg. rank (vs. 6 models)|
|-|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|CARL|69.0|84.4|47.0|26.5|38.9|21.5|21.7|21.7|1.6|

## Data format

- Images: tensors of shape `(B, C, H, W)` where `C` is the spectral dimension.
- Wavelengths: tensors of shape `(B, C)` expressed in **micrometers**.
- Inputs are expected to be mean/std normalized (either global training-set stats or per-image normalization).

Dataset classes live in `carl/data`. The dataset class must have the same name as the corresponding Python file. In the config file, it must look like this:

```yaml
# ...example snippet...
train_dataset:
  name: MyDatasetClass
  # ...dataset args...
```

For supervised training, [GeoBench](https://github.com/ServiceNow/geo-bench) dataset wrappers have been implemented for classification and segmentation.

For self-supervised pretraining, dataset classes for [BigEarthNet](https://bigearth.net/), [SpectralEarth](https://github.com/AABNassim/spectral_earth), and [HySpecNet-11k](https://hyspecnet.rsim.berlin/) have been integrated.
In particular, the training data comes from distinct sensors with different channel counts.
To accommodate this, a custom dataloader is utilized, which can be found in `carl/data/dataloader.py`.

## Project structure (high level)

- `carl/` — datasets, model, modules and trainers
- `segmentation_heads/` — segmentation heads such as ViT-Adapter, UperNet, Mask2Former
- `configs/` — example YAML configs for training and evaluation
- `example.py` — minimal script showing model loading and feature extraction
- `main_*.py` — training/evaluation entry points (Lightning)

## License and third-party code

Please review the root `LICENSE` file for full license terms. Third-party code is licensed under `LICENSE-THIRD-PARTY` or Apache 2.0 as in `LICENSE`.

## Citation

If you use CARL, please cite:

```bibtex
@inproceedings{
baumann2026carl,
title={{CARL}: Camera-Agnostic Representation Learning for Spectral Image Analysis},
author={Alexander Baumann and Leonardo Ayala and Silvia Seidlitz and Jan Sellner and Alexander Studier-Fischer and Berkin {\"O}zdemir and Lena Maier-hein and Slobodan Ilic},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=TpbhS1yfz0}
}
```

## Funding

This project has received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (grant agreement No. 101002198).

![ERC](readme_images/LOGO_ERC-FLAG_EU_.jpg "ERC")
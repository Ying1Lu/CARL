"""CARL SSL checkpoint付き推論テスト
事前学習済み重みをロードして7bandデータで特徴抽出。
"""
import sys
import os
import yaml
import torch
import numpy as np

sys.path.insert(0, ".")
from carl.trainer.seg_trainer import LinearTrainer


def main():
    base_path = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(base_path, "configs", "config_seg.yaml")
    ckpt_path = os.path.join(base_path, "carl_ssl_checkpoint.ckpt")
    
    print("=" * 60)
    print("CARL SSL Checkpoint Load + Inference Test")
    print("=" * 60)
    
    # --- Config 読み込み ---
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # SSL checkpoint path を更新
    cfg["training_kwargs"]["ssl_ckpt_path"] = ckpt_path
    
    print(f"\n[1] Loading model with SSL checkpoint...")
    print(f"    Checkpoint: {ckpt_path}")
    print(f"    Size: {os.path.getsize(ckpt_path) / 1024**2:.1f} MB")
    
    # --- チェックポイント読み込み ---
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    print(f"    Checkpoint keys: {list(checkpoint.keys())}")
    
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        print(f"    State dict keys (first 10): {list(state_dict.keys())[:10]}")
        print(f"    Total keys: {len(state_dict)}")
    
    # --- モデル構築 & 重みロード ---
    print(f"\n[2] Building LinearTrainer model...")
    model = LinearTrainer(cfg)
    
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"    Missing keys:    {len(missing)}")
    print(f"    Unexpected keys: {len(unexpected)}")
    if missing:
        print(f"    (Missing examples: {missing[:5]})")
    if unexpected:
        print(f"    (Unexpected examples: {unexpected[:5]})")
    
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device, dtype=torch.float32)
    print(f"    Device: {device}")
    
    # --- 7band 推論テスト ---
    print(f"\n[3] Inference with synthetic 7-band data...")
    B, C, H, W = 1, 7, 128, 128
    img = torch.randn(B, C, H, W, device=device)
    img = (img - img.mean()) / img.std()  # normalize
    
    # MultiCFカメラの中心波長 (μm)
    wavelengths_um = torch.tensor([[0.47, 0.51, 0.55, 0.59, 0.63, 0.66, 0.70]], 
                                   device=device)
    
    print(f"    Input:       ({B}, {C}, {H}, {W})")
    print(f"    Wavelengths: {wavelengths_um[0].tolist()} μm")
    
    with torch.no_grad():
        spatial_repr, spectral_repr = model.model(img, wavelengths_um)
    
    print(f"    Spatial out:  {spatial_repr.shape}")
    print(f"    Spectral out: {spectral_repr.shape}")
    
    # --- 33band 推論テスト ---
    print(f"\n[4] Inference with synthetic 33-band data...")
    C2 = 33
    img2 = torch.randn(1, C2, H, W, device=device)
    img2 = (img2 - img2.mean()) / img2.std()
    wavelengths_33 = torch.linspace(0.47, 0.70, C2, device=device).unsqueeze(0)
    
    with torch.no_grad():
        spatial_repr2, spectral_repr2 = model.model(img2, wavelengths_33)
    
    print(f"    Input:       (1, {C2}, {H}, {W})")
    print(f"    Spatial out:  {spatial_repr2.shape}")
    print(f"    Spectral out: {spectral_repr2.shape}")
    
    # --- 特徴の統計 ---
    print(f"\n[5] Feature statistics (7-band, pretrained):")
    print(f"    Spatial  - mean: {spatial_repr.mean():.4f}, std: {spatial_repr.std():.4f}")
    print(f"    Spectral - mean: {spectral_repr.mean():.4f}, std: {spectral_repr.std():.4f}")
    
    if torch.cuda.is_available():
        mem_mb = torch.cuda.max_memory_allocated() / 1024**2
        print(f"\n    Peak GPU memory: {mem_mb:.0f} MB")
    
    print("\n" + "=" * 60)
    print("✅ SSL checkpoint loaded and inference successful!")
    print("=" * 60)


if __name__ == "__main__":
    main()

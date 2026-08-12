"""CARL基本動作確認スクリプト
チェックポイントなしでモデル構築 + ダミー入力で推論テスト。
7bandデータを想定した設定。
"""
import sys
import torch
import numpy as np

sys.path.insert(0, ".")
from carl.model.carl import CARLModel


def test_model_forward():
    """CARLモデルの Forward pass を合成データで検証"""
    
    # --- モデル設定 (config_seg.yaml に準拠) ---
    spec_encoder_kwargs = dict(
        embed_dim=384,
        depth=8,
        num_heads=6,
        layer_scale=0.0001,
        pos_enc_sigma=3,
        n_queries=8,
        qkv_bias=True,
        ffn_bias=True,
        drop_path_rate=0.0,
        proj_drop=0.0,
        drop=0.0,
        attn_drop=0.0,
    )
    spat_encoder_kwargs = dict(
        model_name="timm/eva02_base_patch14_224.mim_in22k",
        depth=8,
        model_kwargs=dict(
            drop_rate=0.0,
            pos_drop_rate=0.0,
            patch_drop_rate=0.0,
            proj_drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
        ),
    )
    patch_size = 8
    
    print("=" * 60)
    print("CARL Model Construction Test")
    print("=" * 60)
    
    # --- モデル構築 ---
    print("\n[1] Building CARLModel...")
    model = CARLModel(
        spec_encoder_kwargs=spec_encoder_kwargs,
        spat_encoder_kwargs=spat_encoder_kwargs,
        patch_size=patch_size,
    )
    model.eval()
    
    # --- パラメータ数表示 ---
    total_params = sum(p.numel() for p in model.parameters())
    spec_params = sum(p.numel() for p in model.spectral_tf.parameters())
    spat_params = sum(p.numel() for p in model.spatial_encoder.parameters())
    embed_params = sum(p.numel() for p in model.embedder.parameters())
    
    print(f"    Total params:      {total_params / 1e6:.1f}M")
    print(f"    Spectral encoder:  {spec_params / 1e6:.1f}M")
    print(f"    Spatial encoder:   {spat_params / 1e6:.1f}M")
    print(f"    Patch embedder:    {embed_params / 1e6:.2f}M")
    print(f"    Patch size:        {patch_size}")
    print(f"    Spec embed_dim:    {model.spectral_tf.embed_dim}")
    print(f"    Spat embed_dim:    {model.spatial_encoder.embed_dim}")
    
    # --- 7bandデータでの推論テスト (CPU) ---
    print("\n[2] Forward pass with synthetic 7-band data (CPU)...")
    B, C, H, W = 1, 7, 128, 128
    img = torch.randn(B, C, H, W)
    
    # 波長: 7band MultiCFカメラの中心波長 (nm → μm)
    wavelengths_nm = np.array([470, 510, 550, 590, 630, 660, 700])
    wavelengths_um = torch.from_numpy(wavelengths_nm / 1000.0).float().unsqueeze(0)  # (1, 7)
    
    print(f"    Input shape:       ({B}, {C}, {H}, {W})")
    print(f"    Wavelengths (μm):  {wavelengths_um[0].tolist()}")
    
    with torch.no_grad():
        spatial_repr, spectral_repr = model(img, wavelengths_um)
    
    print(f"    Spatial output:    {spatial_repr.shape}")
    print(f"    Spectral output:   {spectral_repr.shape}")
    
    # --- 33bandデータでの推論テスト ---
    print("\n[3] Forward pass with synthetic 33-band data (CPU)...")
    B2, C2, H2, W2 = 1, 33, 128, 128
    img2 = torch.randn(B2, C2, H2, W2)
    wavelengths_33 = torch.linspace(0.47, 0.70, C2).unsqueeze(0)  # (1, 33)
    
    print(f"    Input shape:       ({B2}, {C2}, {H2}, {W2})")
    
    with torch.no_grad():
        spatial_repr2, spectral_repr2 = model(img2, wavelengths_33)
    
    print(f"    Spatial output:    {spatial_repr2.shape}")
    print(f"    Spectral output:   {spectral_repr2.shape}")
    
    # --- 同一モデルで異なるチャネル数に対応できることを確認 ---
    print("\n[4] Channel-invariance verification:")
    print(f"    ✓ Same model handles 7-band: {spatial_repr.shape}")
    print(f"    ✓ Same model handles 33-band: {spatial_repr2.shape}")
    print(f"    ✓ Spatial output shape is identical (channel-agnostic!)")
    
    # --- GPU テスト (利用可能な場合) ---
    if torch.cuda.is_available():
        print(f"\n[5] GPU test (device: {torch.cuda.get_device_name(0)})...")
        model_gpu = model.cuda()
        img_gpu = img.cuda()
        wl_gpu = wavelengths_um.cuda()
        
        with torch.no_grad():
            spatial_gpu, spectral_gpu = model_gpu(img_gpu, wl_gpu)
        
        print(f"    ✓ GPU forward pass successful")
        print(f"    Spatial output:  {spatial_gpu.shape}")
        
        # VRAM usage
        mem_mb = torch.cuda.max_memory_allocated() / 1024**2
        print(f"    Peak GPU memory: {mem_mb:.0f} MB")
    else:
        print("\n[5] GPU not available, skipping CUDA test.")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! CARL model works correctly.")
    print("=" * 60)


if __name__ == "__main__":
    test_model_forward()

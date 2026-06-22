import torch
import torch.nn.functional as F
from models import UNet, AdaptiveScaleFusion3D
from masking import SpatiallyAdaptiveMultiScaleMasker3D


def train_minimal_pipeline():
    """A minimal demonstration of the SAM-B2B forward and backward process."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Initialize Core Modules
    widths = [2, 3, 4]
    masker = SpatiallyAdaptiveMultiScaleMasker3D(widths=widths).to(device)
    unet = UNet(in_channels=1, out_channels=1).to(device)
    fusion = AdaptiveScaleFusion3D(n_scales=len(widths)).to(device)

    optimizer = torch.optim.Adam(list(unet.parameters()) + list(fusion.parameters()), lr=1e-4)

    # 2. Dummy Data (Batch=2, Channels=1, Depth=32, Height=64, Width=64)
    # In practice, replace this with your DataLoader
    clean_data = torch.rand(2, 1, 32, 64, 64).to(device)
    noisy_data = clean_data + torch.randn_like(clean_data) * 0.1  # Add simple noise

    print("Starting minimal training step...")

    # ==========================
    # CORE PIPELINE
    # ==========================
    optimizer.zero_grad()

    # Step A: Adaptive Masking (Break spatial correlation)
    net_input_all, masks_per_scale, scale_info = masker(noisy_data)

    # Step B: Denoising via 3D U-Net
    # Shape of net_input_all is (Total_Masks, C, D, H, W)
    out_all = unet(net_input_all)

    # Step C: Reorganize outputs back to their respective scales
    N, C, D, H, W = noisy_data.shape
    preds_per_scale = []
    start = 0

    for scale_idx, M, batch_size in scale_info:
        if batch_size == 0: continue
        out_scale = out_all[start:start + batch_size].view(N, M, C, D, H, W)

        # Aggregate M predictions using mask weights
        masks_s = masks_per_scale[scale_idx]
        weights = masks_s.float() / (masks_s.float().sum(dim=1, keepdim=True) + 1e-8)
        aggregated = (out_scale * weights).sum(dim=1)

        preds_per_scale.append(aggregated)
        start += batch_size

    # Step D: Adaptive Multi-scale Fusion
    final_denoised = fusion(preds_per_scale)

    # Step E: Self-Supervised MSE Loss (Target is the noisy data itself!)
    loss = F.mse_loss(final_denoised, noisy_data)
    loss.backward()
    optimizer.step()

    print(f"Training step successful! Loss: {loss.item():.4f}")


if __name__ == '__main__':
    train_minimal_pipeline()
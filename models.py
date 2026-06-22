import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 1. 3D U-Net Backbone
# ==========================================
class LR(nn.Module):
    def __init__(self, in_size, out_size, ksize=3, slope=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_size, out_size, kernel_size=ksize, padding=ksize // 2),
            nn.LeakyReLU(slope, inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UP(nn.Module):
    def __init__(self, in_size, out_size, slope=0.1):
        super().__init__()
        self.conv_1 = LR(in_size, out_size)
        self.conv_2 = LR(out_size, out_size)

    def forward(self, x, pool):
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
        x = torch.cat([x, pool], 1)
        return self.conv_2(self.conv_1(x))


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, depth=4, wf=32, slope=0.1):
        super().__init__()
        self.head = nn.Sequential(LR(in_channels, wf, 3, slope), LR(wf, wf, 3, slope))
        self.down_path = nn.ModuleList([LR(wf, wf, 3, slope) for _ in range(depth)])
        self.up_path = nn.ModuleList()
        for i in range(depth):
            in_c = wf * 2 if i == 0 else wf * 3
            if i == depth - 1: in_c = wf * 2 + in_channels
            self.up_path.append(UP(in_c, wf * 2, slope))
        self.last = nn.Sequential(LR(2 * wf, 2 * wf, 1, slope), LR(2 * wf, 2 * wf, 1, slope),
                                  nn.Conv3d(2 * wf, out_channels, kernel_size=1))

    def forward(self, x):
        blocks = [x]
        x = self.head(x)
        for i, down in enumerate(self.down_path):
            x = F.max_pool3d(x, 2)
            if i != len(self.down_path) - 1: blocks.append(x)
            x = down(x)
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])
        return self.last(x)


# ==========================================
# 2. Adaptive Scale Fusion (Core Contribution)
# ==========================================
class AdaptiveScaleFusion3D(nn.Module):
    """Integrates multi-scale predictions using attention and residual fusion."""

    def __init__(self, n_scales, in_ch=1, hidden=32, fusion_strength=0.05):
        super().__init__()
        self.fusion_strength = fusion_strength
        self.attention_net = nn.Sequential(
            nn.Conv3d(n_scales * in_ch, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, n_scales, kernel_size=1)
        )

    def forward(self, preds_per_scale):
        # preds_per_scale: List of tensors, each (N, C, D, H, W)
        if len(preds_per_scale) == 1:
            return preds_per_scale[0]

        stacked_preds = torch.stack(preds_per_scale, dim=1)  # (N, scales, C, D, H, W)

        # 1. Hard Selection (Sum, as spatial regions are mutually exclusive)
        hard_fused = stacked_preds.sum(dim=1)

        # 2. Soft Attention Integration
        concat_preds = torch.cat(preds_per_scale, dim=1)  # (N, scales*C, D, H, W)
        attn_weights = F.softmax(self.attention_net(concat_preds), dim=1).unsqueeze(2)
        soft_fused = (stacked_preds * attn_weights).sum(dim=1)

        # 3. Residual Fusion
        residual = soft_fused - hard_fused
        out = hard_fused + self.fusion_strength * residual

        return out
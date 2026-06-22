import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 1. 3D Spatial Complexity Analysis
# ==========================================
class ComplexityAnalyzer3D(nn.Module):
    """Calculates 3D spatial gradient to determine local texture complexity."""

    def __init__(self):
        super().__init__()
        # 3D Sobel Operators
        sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
                                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32).view(1, 1, 3, 3, 3)
        sobel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                [[-2, -4, -2], [0, 0, 0], [2, 4, 2]],
                                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32).view(1, 1, 3, 3, 3)
        sobel_z = torch.tensor([[[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]],
                                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                [[1, 2, 1], [2, 4, 2], [1, 2, 1]]], dtype=torch.float32).view(1, 1, 3, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        self.register_buffer('sobel_z', sobel_z)

    def forward(self, x):
        N, C, D, H, W = x.shape
        x_flat = x.view(N * C, 1, D, H, W)
        x_scaled = x_flat / (torch.std(x_flat) + 1e-8)

        grad_x = F.conv3d(x_scaled, self.sobel_x, padding=1)
        grad_y = F.conv3d(x_scaled, self.sobel_y, padding=1)
        grad_z = F.conv3d(x_scaled, self.sobel_z, padding=1)

        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + grad_z ** 2 + 1e-8)
        return grad_mag.view(N, C, D, H, W)


# ==========================================
# 2. Adaptive Mask Generation
# ==========================================
class SpatiallyAdaptiveMultiScaleMasker3D(nn.Module):
    """Assigns different blind-spot scales based on complexity."""

    def __init__(self, widths=[2, 3, 4], flat_ratio=0.6, edge_ratio=0.1):
        super().__init__()
        self.widths = widths
        self.flat_ratio = flat_ratio  # Ratio of flat regions (largest scale)
        self.edge_ratio = edge_ratio  # Ratio of complex edges (smallest scale)
        self.analyzer = ComplexityAnalyzer3D()

    def _make_fix_mask_3d(self, shape, width, idx, device):
        mask = torch.zeros(shape, device=device, dtype=torch.bool)
        dz, dy, dx = idx // (width * width), (idx // width) % width, idx % width
        mask[:, :, dz::width, dy::width, dx::width] = True
        return mask

    def _interpolate_mask(self, tensor, mask_bool):
        n, c, d, h, w = tensor.shape
        kernel = torch.ones((1, 1, 3, 3, 3), device=tensor.device, dtype=tensor.dtype)
        kernel[0, 0, 1, 1, 1] = 0.0
        kernel = kernel / kernel.sum()
        filt = F.conv3d(tensor.view(n * c, 1, d, h, w), kernel, stride=1, padding=1).view(n, c, d, h, w)
        return torch.where(mask_bool, filt, tensor)

    def forward(self, img):
        N, C, D, H, W = img.shape
        device = img.device

        # 1. Analyze complexity & allocate scales
        complexity_score = self.analyzer(img).mean(dim=1)
        scale_map = torch.zeros(N, D, H, W, dtype=torch.long, device=device)

        for n in range(N):
            comp_flat = complexity_score[n].view(-1)
            thresh_flat = torch.quantile(comp_flat, self.flat_ratio)
            thresh_edge = torch.quantile(comp_flat, 1.0 - self.edge_ratio)

            scale_map[n][complexity_score[n] >= thresh_edge] = 0  # Complex -> Width 2
            scale_map[n][
                (complexity_score[n] > thresh_flat) & (complexity_score[n] < thresh_edge)] = 1  # Mid -> Width 3
            scale_map[n][complexity_score[n] <= thresh_flat] = 2  # Flat -> Width 4

        # 2. Generate masked inputs per scale
        inputs_per_scale, masks_per_scale, scale_info, all_inputs = [], [], [], []

        for scale_idx, width in enumerate(self.widths):
            scale_mask = (scale_map == scale_idx)
            M = width ** 3
            scale_tensors, scale_masks = [], []

            for mask_idx in range(M):
                bs_mask = self._make_fix_mask_3d(img.shape, width, mask_idx, device)
                combined_mask = bs_mask & scale_mask.unsqueeze(1)
                masked_img = self._interpolate_mask(img, combined_mask)

                scale_tensors.append(masked_img)
                scale_masks.append(combined_mask)

            stacked_tensors = torch.stack(scale_tensors, dim=1)
            inputs_per_scale.append(stacked_tensors)
            masks_per_scale.append(torch.stack(scale_masks, dim=1))
            all_inputs.append(stacked_tensors.view(N * M, C, D, H, W))
            scale_info.append((scale_idx, M, N * M))

        net_input_all = torch.cat(all_inputs, dim=0)
        return net_input_all, masks_per_scale, scale_info
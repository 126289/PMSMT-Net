import torch
import torch.nn as nn
import torch.nn.functional as F

# 视觉感知模块
class MSDCBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(MSDCBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=3, dilation=3)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        out = self.conv1(x) + self.conv2(x) + self.conv3(x)
        return self.relu(self.bn(out))

class RobertsEdge(nn.Module):
    def __init__(self):
        super(RobertsEdge, self).__init__()
        kernel_x = torch.tensor([[1, 0], [0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        kernel_y = torch.tensor([[0, 1], [-1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer('weight_x', kernel_x)
        self.register_buffer('weight_y', kernel_y)

    def forward(self, x):
        edge_x = F.conv2d(x, self.weight_x, padding=0)
        edge_y = F.conv2d(x, self.weight_y, padding=0)
        edge = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)
        return edge


# ---- Approximate 2D FRFT via Frequency Mask ----
class FRFT2D(nn.Module):
    def __init__(self):
        super(FRFT2D, self).__init__()

    def forward(self, x):
        fft = torch.fft.fft2(x)
        amp = torch.abs(fft)
        amp = amp / (torch.max(amp) + 1e-8)
        return amp.unsqueeze(1)  # add channel dim

# ---- VPM Block ----
class VPMBlock(nn.Module):
    def __init__(self, in_channels):
        super(VPMBlock, self).__init__()
        self.edge_extractor = RobertsEdge()
        self.frft_extractor = FRFT2D()

        self.edge_conv = nn.Sequential(
            nn.Conv2d(1, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.freq_conv = nn.Sequential(
            nn.Conv2d(1, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 原始特征图 x: (B, C, H, W)
        x_mean = torch.mean(x, dim=1, keepdim=True)  # convert to grayscale

        edge_map = self.edge_extractor(x_mean)
        edge_feat = self.edge_conv(edge_map)

        freq_map = self.frft_extractor(x_mean)
        freq_feat = self.freq_conv(freq_map)

        x_fuse = torch.cat([x, edge_feat, freq_feat], dim=1)
        out = self.fusion_conv(x_fuse)
        return out + x  # residual connection


# 残差 + CBAM 注意力模块
class VR_CBAMBlock(nn.Module):
    def __init__(self, channels):
        super(VR_CBAMBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid()
        )
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        res = x
        x = self.relu(self.conv(x))
        ca_out = self.ca(x) * x

        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        sa_input = torch.cat([avg_pool, max_pool], dim=1)
        sa_out = self.sa(sa_input) * ca_out

        return sa_out + res


# 解码模块：RFB + 可分离卷积
class RFB_PSCBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(RFB_PSCBlock, self).__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

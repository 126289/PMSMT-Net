import torch
import torch.nn as nn
import torch.nn.functional as F

# 多尺度空洞卷积块
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


# 视觉感知模块（融合边缘和频域信息）
class VPMBlock(nn.Module):
    def __init__(self, channels):
        super(VPMBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention = self.conv(x)
        return x * attention


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

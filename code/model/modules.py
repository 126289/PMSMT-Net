import torch
import torch.nn as nn
import torch.nn.functional as F

# MSDC+VPM
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

class FRFT2D(nn.Module):
    def __init__(self):
        super(FRFT2D, self).__init__()

    def forward(self, x):
        fft = torch.fft.fft2(x)
        amp = torch.abs(fft)
        amp = amp / (torch.max(amp) + 1e-8)
        return amp.unsqueeze(1)  # add channel dim

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
        x_mean = torch.mean(x, dim=1, keepdim=True)  # convert to grayscale

        edge_map = self.edge_extractor(x_mean)
        edge_feat = self.edge_conv(edge_map)

        freq_map = self.frft_extractor(x_mean)
        freq_feat = self.freq_conv(freq_map)

        x_fuse = torch.cat([x, edge_feat, freq_feat], dim=1)
        out = self.fusion_conv(x_fuse)
        return out + x  # residual connection
class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(EncoderBlock, self).__init__()
        self.msdc = MSDCBlock(in_ch, out_ch)
        self.vpm = VPMBlock(out_ch)

    def forward(self, x):
        x = self.msdc(x)
        x = self.vpm(x)
        return x

# VR_CBAMBlock
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) 
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # shape: (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))


class VR_CBAMBlock(nn.Module):
    def __init__(self, channels, num_res_blocks=2):
        super(VR_CBAMBlock, self).__init__()

        self.res_blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_res_blocks)])
        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, Tn, Tn_plus1=None):
        x = self.res_blocks(Tn) 
        if Tn_plus1 is not None:
            x = torch.cat([x, Tn_plus1], dim=1)
            x = self.fusion_conv(x)
        ca = self.channel_attention(x)
        sa = self.spatial_attention(x)
        x = ca * x
        x = sa * x
        return x


# RFB_PSCBlock
class RFB_PSCBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RFB_PSCBlock, self).__init__()
        inter_channels = out_channels // 4
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels),  # depthwise
            nn.Conv2d(out_channels, out_channels, kernel_size=1),  # pointwise
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        out = torch.cat([b1, b2, b3, b4], dim=1)  # concat on channel
        out = self.fusion(out)
        return out

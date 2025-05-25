import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import MSDCBlock, VPMBlock, VR_CBAMBlock, RFB_PSCBlock
from classifier import MultiInputClassifier

class PMSMTNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=3, base_channels=64):
        super(PMSMTNet, self).__init__()

        # --------- Encoder blocks ---------
        self.enc1 = MSDCBlock(in_channels, base_channels)
        self.vpm1 = VPMBlock(base_channels)

        self.enc2 = MSDCBlock(base_channels, base_channels*2)
        self.vpm2 = VPMBlock(base_channels*2)

        self.enc3 = MSDCBlock(base_channels*2, base_channels*4)
        self.vpm3 = VPMBlock(base_channels*4)

        self.enc4 = MSDCBlock(base_channels*4, base_channels*8)
        self.vpm4 = VPMBlock(base_channels*8)

        # --------- Bottleneck ---------
        self.bottleneck = MSDCBlock(base_channels*8, base_channels*16)

        # --------- Decoder blocks ---------
        self.up4 = RFB_PSCBlock(base_channels*16, base_channels*8)
        self.up3 = RFB_PSCBlock(base_channels*8, base_channels*4)
        self.up2 = RFB_PSCBlock(base_channels*4, base_channels*2)
        self.up1 = RFB_PSCBlock(base_channels*2, base_channels)

        # --------- Skip Connection Attention ---------
        self.cbam4 = VR_CBAMBlock(base_channels*8)
        self.cbam3 = VR_CBAMBlock(base_channels*4)
        self.cbam2 = VR_CBAMBlock(base_channels*2)
        self.cbam1 = VR_CBAMBlock(base_channels)

        # --------- Segmentation head ---------
        self.seg_out = nn.Conv2d(base_channels, 1, kernel_size=1)

        # --------- Classification branch ---------
        self.classifier = MultiInputClassifier(num_classes=num_classes)

    def forward(self, x, edge_img=None, morph_img=None):
        # --------- Encoder path ---------
        e1 = self.vpm1(self.enc1(x))    # out: C
        e2 = self.vpm2(self.enc2(F.max_pool2d(e1, 2)))  # out: 2C
        e3 = self.vpm3(self.enc3(F.max_pool2d(e2, 2)))  # out: 4C
        e4 = self.vpm4(self.enc4(F.max_pool2d(e3, 2)))  # out: 8C

        # --------- Bottleneck ---------
        bn = self.bottleneck(F.max_pool2d(e4, 2))       # out: 16C

        # --------- Decoder path ---------
        d4 = self.up4(bn) + self.cbam4(e4)
        d3 = self.up3(d4) + self.cbam3(e3)
        d2 = self.up2(d3) + self.cbam2(e2)
        d1 = self.up1(d2) + self.cbam1(e1)

        # --------- Segmentation output ---------
        seg_mask = torch.sigmoid(self.seg_out(d1))

        # --------- Classification (multi-input) ---------
        cls_out = self.classifier(seg_mask, edge_img, morph_img)

        return seg_mask, cls_out

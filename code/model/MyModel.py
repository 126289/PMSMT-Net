import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, in_channels=1, num_classes=3, base_channels=64, dropout_p=0.3):
        super(MyModel, self).__init__()

        self.dropout_p = dropout_p
        self.dropout = nn.Dropout2d(p=dropout_p)

        self.enc1 = EncoderBlock(in_channels, base_channels)
        self.enc2 = EncoderBlock(base_channels, base_channels*2)
        self.enc3 = EncoderBlock(base_channels*2, base_channels*4)
        self.enc4 = EncoderBlock(base_channels*4, base_channels*8)

        self.bottleneck = MSDCBlock(base_channels*8, base_channels*16)

        self.up4 = RFB_PSCBlock(base_channels*16, base_channels*8)
        self.up3 = RFB_PSCBlock(base_channels*8, base_channels*4)
        self.up2 = RFB_PSCBlock(base_channels*4, base_channels*2)
        self.up1 = RFB_PSCBlock(base_channels*2, base_channels)
   
        self.cbam4 = VR_CBAMBlock(base_channels*8, num_res_blocks=4)
        self.cbam3 = VR_CBAMBlock(base_channels*4, num_res_blocks=3)
        self.cbam2 = VR_CBAMBlock(base_channels*2, num_res_blocks=2)
        self.cbam1 = VR_CBAMBlock(base_channels, num_res_blocks=1)

        self.seg_out = nn.Conv2d(base_channels, 1, kernel_size=1)

        self.classifier = EnsembleClassifier(num_classes=num_classes)

    def forward(self, x, edge_img=None, morph_img=None):
        # Encoder path
        e1 = self.enc1(x)
        e1 = self.dropout(e1)

        e2 = self.enc2(F.max_pool2d(e1, 2))
        e2 = self.dropout(e2)

        e3 = self.enc3(F.max_pool2d(e2, 2))
        e3 = self.dropout(e3)

        e4 = self.enc4(F.max_pool2d(e3, 2))
        e4 = self.dropout(e4)

        bn = self.bottleneck(F.max_pool2d(e4, 2))
        bn = self.dropout(bn)

        d4 = self.up4(bn)
        d4 = d4 + self.cbam4(e4)
        d4 = self.dropout(d4)

        d3 = self.up3(d4)
        d3 = d3 + self.cbam3(e3)
        d3 = self.dropout(d3)

        d2 = self.up2(d3)
        d2 = d2 + self.cbam2(e2)
        d2 = self.dropout(d2)

        d1 = self.up1(d2)
        d1 = d1 + self.cbam1(e1)
        d1 = self.dropout(d1)

        # Segmentation output
        seg_mask = torch.sigmoid(self.seg_out(d1))

        # Classification
        cls_out = self.classifier(seg_mask, edge_img, morph_img)

        return seg_mask, cls_out
if __name__ == '__main__':
    from torch.autograd import Variable
    x = Variable(torch.rand(2, 3, 256, 256)).cuda()
    model = MyModel().cuda()
    c, s = model(x)
    print('Output s shape:', s.shape)
    print('Output c shape:', c.shape)

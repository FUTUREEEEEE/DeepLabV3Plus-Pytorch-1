# -*- coding: utf-8 -*-
import torch.nn as nn

class GAU(nn.Module):
    def __init__(self):
        super(GAU, self).__init__()
        # Global Attention Upsample
        #self.upsample = upsample
        # self.conv3x3 = nn.Conv2d(channels_low, channels_low, kernel_size=3, padding=1, bias=False)
        # self.bn_low = nn.BatchNorm2d(channels_low)

        self.conv1x1 = nn.Conv2d(256, 256, kernel_size=1, padding=0, bias=False)
        self.bn_high = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        
        # if upsample:
        #     self.conv_upsample = nn.ConvTranspose2d(channels_high, channels_low, kernel_size=4, stride=2, padding=1, bias=False)
        #     self.bn_upsample = nn.BatchNorm2d(channels_low)
        # else:
        #     self.conv_reduction = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
        #     self.bn_reduction = nn.BatchNorm2d(channels_low)
        # self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        """
        Use the high level features with abundant catagory information to weight the low level features with pixel
        localization information. In the meantime, we further use mask feature maps with catagory-specific information
        to localize the mask position.
        :param fms_high: Features of high level. Tensor.
        :param fms_low: Features of low level.  Tensor.
        :param fm_mask:
        :return: fms_att_upsample
        """
        b, c, h, w = x.shape

        x = nn.AvgPool2d(x.shape[2:])(x).view(len(x), c, 1, 1)
        x = self.conv1x1(x)
        x = self.bn_high(x)
        out = self.relu(x)
            
        # fms_low_mask = torch.cat([fms_low, fm_mask], dim=1)
        # fms_low_mask = self.conv3x3(fms_low)
        # fms_low_mask = self.bn_low(fms_low_mask)

        # fms_att = fms_low_mask * fms_high_gp
        # if self.upsample:
        #     out = self.relu(
        #         self.bn_upsample(self.conv_upsample(fms_high)) + fms_att)
        # else:
        #     out = self.relu(
        #         self.bn_reduction(self.conv_reduction(fms_high)) + fms_att)

        return out
# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import functional as F

class FPN_Module(nn.Module):

    
    """
    input  feature['low_level']  feature['feat2']   feature['feat3']
    
    out   featur low_level_feature
    
    ('low_level', 1/4 torch.Size([1, 256, 128, 128])), ('feat2', 1/8 torch.Size([1, 512, 64, 64])), 
    ('feat3', 1/16 torch.Size([1, 1024, 32, 32])), ('out', torch.Size([1, 2048, 32, 32]))
    
    """
    def __init__(self):
        super(FPN_Module, self).__init__()
        
        self.project = nn.Sequential( 
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        
        
        self.convfeat3 = nn.Sequential( 
            nn.Conv2d(1024, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.convfeat2= nn.Sequential( 
            nn.Conv2d(512, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        
        
    def forward(self,feature):
        
        feat3=F.interpolate(feature['feat3'], scale_factor=2, mode='bilinear', align_corners=False)
        feat3=self.convfeat3(feat3)
        
        feat2=feature['feat2'] +feat3
        feat2=F.interpolate(feat2, scale_factor=2, mode='bilinear', align_corners=False)
        feat2=self.convfeat2(feat2)
        
        
        feat1=feature['low_level']+feat2
        feat1=self.project(feat1)
        

    
        
        
        return feat1   #48*128*128
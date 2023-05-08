# author: JingXiaolun
# date: 2022.04.06
# description: SE Block(squeeze and excitation)
# paper: https://arxiv.org/pdf/1709.01507.pdf(Squeeze-and-Excitation Networks)

import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, frame_length, reduction_ratio):
        super().__init__()
        '''
        frame_length: length of extracted frame embeddings
        reduction_ratio: ratio used to reduce the length of video frame to constrcut nonlinear relations between different frames
        '''
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # reducetion + expansion
        self.fc = nn.Sequential(
                nn.Linear(frame_length, frame_length // reduction_ratio),
                nn.ReLU(inplace=True),
                nn.Linear(frame_length // reduction_ratio, frame_length),
                nn.Sigmoid()
                )
        
        ## expansion + reduction
        #self.fc = nn.Sequential(
        #        nn.Linear(frame_length, frame_length * reduction_ratio),
        #        nn.ReLU(inplace=True),
        #        nn.Linear(frame_length * reduction_ratio, frame_length),
        #        nn.Sigmoid()
        #        )

    def forward(self, x):
        b, n = x.shape[:2]
        # step1: squeeze 
        weight = self.avgpool(x).contiguous().view(b, -1)
        # step2: excitation
        weight = self.fc(weight).contiguous().view(b, n, 1)
        # step3: scale
        out = torch.mul(x, weight)
        return out

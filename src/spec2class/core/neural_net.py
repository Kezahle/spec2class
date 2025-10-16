
import torch
import numpy as np
import torch.nn as nn
from torch.nn import Conv1d,ReLU, MaxPool1d, Linear, Dropout, Sigmoid
#Here my input vector will have 5000 bins


class Net(nn.Module):
    def __init__(self,bin_no, drop_conv, drop_linear): # I need to understand where do I put the bin number input, actualy i don't think I need it
        super().__init__()
        if drop_conv == None:
            self.drop_conv = 0.2
        if drop_linear == None:
            self.drop_linear = 0.3

        self.drop_conv = drop_conv
        self.drop_linear = drop_linear
        self.bin_no = bin_no

        
       
        self.features = nn.Sequential(
            
           #in channel  = 1  (for rgb image, number of chanels is three)
           #in Length = bin_number
           #kernel is one dimensional and I descide it's length
           #padding  = 0 as I undestand from the original model
           Conv1d(1, 32, kernel_size=3, stride=1, padding=0),
           #Conv1d(1, 32, kernel_size=3, stride=1, padding=0),
           ReLU(inplace=True),
           Conv1d(32, 32, kernel_size=3, stride=1 , padding=0),
           MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
           Dropout(p=drop_conv, inplace=False)
            )

        
        self.classifier = nn.Sequential(
          Linear(in_features= 79936 ,out_features=400 ,bias=True),
          ReLU(inplace=True),
          Linear(in_features=400, out_features=400 ,bias=True),
          ReLU(inplace=True),
          Dropout(p=drop_linear, inplace=False),
          Linear(in_features=400, out_features=1 ,bias=True),
          Sigmoid()
           )

    def forward(self,x):
        
        out = self.features(x)
        out = torch.flatten(out,1)
        #out_length = out.size(1)  #since the zero dimension is the batch size
        out = self.classifier(out)
        
        return out
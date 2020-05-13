########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"
########################################

import torch
import torch.nn as nn

from common.unet import UNetSP
from common.nconv import NConvUNet


class CNN(nn.Module):
    def __init__(self):
        super().__init__() 
        self.__name__ = 'pncnn'

        self.conf_estimator = UNetSP(1, 1)
        self.nconv = NConvUNet(1, 1)
        self.var_estimator = UNetSP(1, 1)

    def forward(self, x0):  
        x0 = x0[:,:1,:,:]  # Use only depth
        c0 = self.conf_estimator(x0)
        xout, cout = self.nconv(x0, c0)
        cout = self.var_estimator(cout)
        out = torch.cat((xout, cout, c0), 1)
        return out
        
       


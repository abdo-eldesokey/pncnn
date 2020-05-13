#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 15:07:02 2019

@author: abdel62
"""
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


class Dataloader(Dataset):

    def __init__(self, dataset, path):
        self.dataset_name = dataset
        self.__dataset_path__ = path
        self.input_paths = []
        self.gt_paths = []
    
    def get_path(self):
        return(self.__dataset_path__)
    
    def set_path(self, path):
        self.__dataset_path__ = path
    
    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, index):
        raise(NotImplementedError)

    def resize_tensor(self, inpt_tensor, sz, mode='bilinear'):
        if isinstance(self.sz, float):
            inpt_tensor = F.interpolate(inpt_tensor.unsqueeze(0), scale_factor=sz, mode=mode)
            return torch.squeeze(inpt_tensor, 0)
        
        elif isinstance(self.sz,tuple):
            inpt_tensor = F.interpolate(inpt_tensor.unsqueeze(0), size=sz, mode=mode)
            return torch.squeeze(inpt_tensor, 0)
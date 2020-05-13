########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"
########################################

from PIL import Image
from torch.utils.data import Dataset
import torch
import numpy as np
import glob
import os
from torchvision import transforms


class KittiDepthDataset(Dataset):

    def __init__(self, kitti_depth_path, setname='train', transform=None, norm_factor=256, invert_depth=False,
                 load_rgb=False, kitti_rgb_path=None, rgb2gray=False, hflip=False):

        self.kitti_depth_path = kitti_depth_path
        self.setname = setname
        self.transform = transform
        self.norm_factor = norm_factor
        self.invert_depth = invert_depth
        self.load_rgb = load_rgb
        self.kitti_rgb_path = kitti_rgb_path
        self.rgb2gray = rgb2gray
        self.hflip = hflip

        if setname in ['train', 'val']:
            depth_path = os.path.join(self.kitti_depth_path, setname)
            self.depth = np.array(sorted(glob.glob(depth_path + "/**/proj_depth/velodyne_raw/**/*.png", recursive=True)))
            self.gt = np.array(sorted(glob.glob(depth_path + "/**/proj_depth/groundtruth/**/*.png", recursive=True)))
        elif setname == 'selval':
            depth_path = os.path.join(self.kitti_depth_path, 'depth_selection', 'val_selection_cropped')
            self.depth = np.array(sorted(glob.glob(depth_path + "/velodyne_raw/*.png", recursive=True)))
            self.gt = np.array(sorted(glob.glob(depth_path + "/groundtruth_depth/*.png", recursive=True)))
        elif setname == 'test':
            depth_path = os.path.join(self.kitti_depth_path, 'depth_selection', 'test_depth_completion_anonymous')
            self.depth = np.array(sorted(glob.glob(depth_path + "/velodyne_raw/*.png", recursive=True)))
            self.gt = np.array(sorted(glob.glob(depth_path + "/velodyne_raw/*.png", recursive=True)))

        assert(len(self.gt) == len(self.depth))

    def __len__(self):
        return len(self.depth)

    def __getitem__(self, item):
        if item < 0 or item >= self.__len__():
            return None

        # Read depth input and gt
        depth = Image.open(self.depth[item])
        gt = Image.open(self.gt[item])

        # Read RGB images 
        if self.load_rgb:
            if self.setname in ['train', 'val']:
                gt_path = self.gt[item]
                idx = gt_path.find('2011')
                seq_name = gt_path[idx:idx+26]
                idx2 = gt_path.find('groundtruth')
                camera_name = gt_path[idx2+12:idx2+20]
                fname = gt_path[idx2+21:]
                rgb_path = os.path.join(self.kitti_rgb_path, self.setname, seq_name, camera_name, 'data', fname)
                rgb = Image.open(rgb_path)
            elif self.setname == 'selval':
                depth_path = self.depth[item]
                tmp = depth_path.split('velodyne_raw')
                rgb_path = tmp[0] + 'image' + tmp[1] + 'image' + tmp[2]
                rgb = Image.open(rgb_path)
            elif self.setname == 'test':
                depth_path = self.depth[item]
                tmp = depth_path.split('velodyne_raw')
                rgb_path = tmp[0] + 'image' + tmp[1]
                rgb = Image.open(rgb_path)

            if self.rgb2gray:
                t = transforms.Grayscale(1)
                rgb = t(rgb)
        
        # Apply transformations if given
        if self.transform is not None:
            depth = self.transform(depth)
            gt = self.transform(gt)
            if self.load_rgb:
                rgb = self.transform(rgb)

        flip_prob = np.random.uniform(0.0, 1.0) > 0.5
        if self.hflip and flip_prob:
            depth, gt = transforms.functional.hflip(depth),  transforms.functional.hflip(gt)
            if self.load_rgb:
                rgb = transforms.functional.hflip(rgb)

        # Convert to numpy
        depth = np.array(depth, dtype=np.float32)
        gt = np.array(gt, dtype=np.float32)

        # Normalize the depth
        depth = depth / self.norm_factor  #[0,1]
        gt = gt / self.norm_factor

        # Expand dims into Pytorch format 
        depth = np.expand_dims(depth, 0)
        gt = np.expand_dims(gt, 0)

        # Convert to Pytorch Tensors
        depth = torch.from_numpy(depth)  #    (depth, dtype=torch.float)
        gt = torch.from_numpy(gt)  #tensor(gt, dtype=torch.float)
        
        # Convert depth to disparity 
        if self.invert_depth:
            depth[depth==0] = -1
            depth = 1 / depth
            depth[depth==-1] = 0

            gt[gt==0] = -1
            gt = 1 / gt
            gt[gt==-1] = 0
        
        # Convert RGB image to tensor
        if self.load_rgb:
            rgb = np.array(rgb, dtype=np.float32)
            rgb /= 255
            if self.rgb2gray: rgb = np.expand_dims(rgb,0)
            else : rgb = np.transpose(rgb,(2,0,1))
            rgb = torch.from_numpy(rgb)
            input = torch.cat((rgb, depth), 0)
        else:
            input = depth

        return input, gt

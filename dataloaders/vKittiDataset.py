import os
import cv2
import numpy as np
import torch
import json
from pathlib import Path
from .vkitti_transforms import Transform, Crop, vkitti_image_transform, vkitti_disparity_transform, \
    vkitti_depth_transform
from torch.utils.data import Dataset


class BufferedDataset:

    def __repr__(self):
        return "Dataset: with " + str(len(self.samples)) + "samples"

    def __init__(self,samples, cache_data_in_ram=False):
        self.cache_data_in_ram = cache_data_in_ram
        self.samples=samples


    def __getitem__(self, index):

        if (index < len(self.samples)):
            if self.cache_data_in_ram:
                return self.samples[index]
            return self.samples[index].copy_paths()
        else:
            raise IndexError

    def __len__(self):
        return len(self.samples)

    # adds all samples from dataset2 to this one...
    def concat(self, dataset2):
        self.samples.extend(dataset2.samples)


class VkittiDatasetWrapper(Dataset):
    # the augmenter here performs transforms which require the entire sample at once,
    # there is also a batch transform
    def __init__(self, samples, load_rgb):
        self.samples = BufferedDataset(samples)
        self.load_rgb = load_rgb

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        if self.load_rgb:
            rgb, d = self.samples[idx].get_images()
            sparse = RandomMasking().transform(d.clone(), idx)
            C = (sparse > 0).float()
            d[d > 254] = 0  # Set points at infinity to zero
            # Crop the groundtruth so it becomes like Kitti
            d[0, :120, :] = 0
            input = torch.cat((rgb, sparse, C), 0)
            return input.to(self.device), d.to(self.device)
        else:
            d = self.samples[idx].get_images()
            sparse = RandomMasking().transform(d.clone(), idx)
            C = (sparse > 0).float()
            d[d > 254] = 0
            d[0, :100, :] = 0
            input = torch.cat((sparse, C), 0)
            return input, d


# Load the KITII masks to use with vKITTI
# TODO: Replace with something cleaner
np.random.seed(0)
random_masking_masks = [] #np.load('/ssd/code/nconv-gen-conf/scripts/generate_input_masks_from_kitti/train_masks_10k.np.npz')['masks']
mask_idxs = np.random.randint(0, len(random_masking_masks), len(random_masking_masks))
class RandomMasking(Transform):
    def __init__(self, transforms=[]):
        super().__init__(transforms)

    def transform(self, img, idx):
        mask_idx = mask_idxs[idx]
        # print(mask_idx)
        x, y = random_masking_masks[mask_idx, :]
        x = x.tolist()
        y = y.tolist()

        zeros = torch.zeros_like(img[0, :, :], dtype=torch.long)
        zeros[x, y] = 1
        zeros = zeros == 0

        img[0, zeros] = 0
        img[img > 254] = 0

        return img


class VKittiDataset:
    # the train,val test split is not very thought trough,

    def __init__(self, basepath, params):
        self.basepath = basepath

        self.training = None
        self.testing = None
        self.validation = None

        self.params = params

    def found(self):
        return self.basepath.exists()

    # returns the samples with paths, not initialized
    def get_training_validation_testing(self):
        assert (len(self.training) + len(self.testing) + len(self.validation) > 0)
        return self.training, self.validation, self.testing,

    def init(self):
        # read all worlds and images

        worlds = ['0001', "0002", "0006", "0018", "0020"]
        cameras = ["15-deg-left", "30-deg-left", "clone", "fog", "morning", "overcast", "rain", "sunset",
                   "15-deg-right", "30-deg-right"]
        images = [447, 232, 269, 338, 836]
        tmp = 0
        for i in images:
            tmp = tmp + i

        total_samples = (len(cameras) * tmp)
        samples = [None] * total_samples
        index = 0

        if self.params['invert_depth']:
            disp_transform = Transform([vkitti_disparity_transform(), Crop(crop_size=[352, 1216])])
        else:
            disp_transform = Transform([vkitti_depth_transform(), Crop(crop_size=[352, 1216])])

        if self.params['load_rgb']:
            im_transform = Transform([vkitti_image_transform(), Crop(crop_size=[352, 1216])])

        for camera in cameras:
            for i, world in enumerate(worlds):
                for sample in range(0, images[i]):
                    disp_path = self.basepath / Path("vkitti_1.3.1_depthgt/") / world / camera / (
                                str(sample).zfill(5) + ".png")
                    if self.params['load_rgb']:
                        rgb_path = self.basepath / Path("vkitti_1.3.1_rgb") / world / camera / (
                                    str(sample).zfill(5) + ".png")
                        samples[index] = VKittiSample(rgb_path, disp_path, im_transform, disp_transform)
                    else:
                        samples[index] = VKittiSampleDepthOnly(disp_path, disp_transform)
                    index += 1

        self.training = samples[0:tmp * 8]
        self.validation = samples[tmp * 8:]
        self.testing = []

        print("Initialized VKitti Dataset")
        print("Training Samples: " + str(len(self.training)))
        print("Validation Samples: " + str(len(self.validation)))
        if len(self.training) == 0:
            quit()

    def __repr__(self):
        tmp = "VKitti: " + str(self.basepath)
        return tmp


class VKittiSampleDepthOnly:
    # provides the VKittiDataset as rgb image and disparity.
    # negative values are missing disparity.
    # warning disparity is caped >0
    def __init__(self, depth_path, disparity_transform):
        self.disparity_path = depth_path
        self.image = None
        self.disparity = None
        self.disparity_transforms = Transform([disparity_transform])

    def copy_paths(self):
        return VKittiSampleDepthOnly(self.disparity_path,
                                     self.disparity_transforms)

    def append_disparity_transform(self, transform):
        self.disparity_transforms.append(transform)

    def get_images(self):
        self.read_from_disk()
        return self.disparity

    def read_image(self, path):
        assert not path is None
        if not Path(str(path)).exists():
            print(str(path))
            assert Path(str(path)).exists()

        return cv2.imread(path, cv2.IMREAD_UNCHANGED)  # cv2.IMREAD_ANYCOLOR| cv2.IMREAD_ANYDEPTH

    def read_from_disk(self):
        if self.disparity is None:
            self.disparity = self.disparity_transforms(self.read_image(str(self.disparity_path)))

    def __repr__(self):
        tmp = "VkittiSample:\n"
        ps = [self.disparity_path]
        for p in ps:
            tmp += str(p) + "\n"
        if (not self.disparity is None):
            tmp += "Images loaded\n"
        else:
            tmp += "Images pending\n"
        return tmp


class VKittiSample:
    # provides the VKittiDataset as rgb image and disparity.
    # negative values are missing disparity.
    # warning disparity is caped >0
    def __init__(self, image_path, depth_path, image_transform, disparity_transform):
        print("image_path: " + str(image_path))
        print("depth_path: " + str(depth_path))
        self.image_path = image_path
        self.disparity_path = depth_path
        self.image = None
        self.disparity = None
        self.image_transforms = Transform([image_transform])
        self.disparity_transforms = Transform([disparity_transform])

    def copy_paths(self):
        return VKittiSample(self.image_path,
                            self.disparity_path,
                            self.image_transforms,
                            self.disparity_transforms)

    def append_image_transform(self, transform):
        self.image_transforms.append(transform)

    def append_disparity_transform(self, transform):
        self.disparity_transforms.append(transform)

    def get_images(self):
        self.read_from_disk()
        return self.image, self.disparity

    def read_image(self, path):
        assert not path is None
        if not Path(str(path)).exists():
            print(str(path))
            assert Path(str(path)).exists()

        return cv2.imread(path, cv2.IMREAD_UNCHANGED)  # cv2.IMREAD_ANYCOLOR| cv2.IMREAD_ANYDEPTH

    def read_from_disk(self):
        if self.image is None:
            self.image = self.image_transforms(self.read_image(str(self.image_path)))
            self.disparity = self.disparity_transforms(self.read_image(str(self.disparity_path)))

    def __repr__(self):
        tmp = "VkittiSample:\n"
        ps = [self.image_path, self.disparity_path]
        for p in ps:
            tmp += str(p) + "\n"
        if (not self.image is None):
            tmp += "Images loaded\n"
        else:
            tmp += "Images pending\n"
        return tmp


def main():
    exp_dir = '/home/abdel62/python_workspace/nconv-gen-conf/workspace/abdo/vkitti/exp_gen_conf'
    # Read parameters file
    with open(os.path.join(exp_dir, 'params.json'), 'r') as fp:
        params = json.load(fp)

    dataloaders, datasets_sizes = get_vkitti(params)
    cv2.namedWindow('Depth')
    cv2.namedWindow('Sparse')
    cv2.namedWindow('RGB')

    for data in dataloaders['val']:
        sparse, C, d, idx = data
        im_d = d[0, 0, :, :].numpy()
        im_sparse = sparse[0, 0, :, :].numpy()
        kernel = np.ones((5, 5), np.uint8)
        im_sparse = cv2.dilate(im_sparse, kernel, 5)
        im_sparse = 255 - im_sparse
        im_sparse[im_sparse == 255] = 0

        # im = C[0,:,:].numpy()
        # im[im>255]=255
        # im[im<0]=0

        # rgb_im = (rgb[0,:,:,:].numpy().transpose(1,2,0)+0.5)*255
        # rgb_im = cv2.cvtColor(rgb_im, cv2.COLOR_RGB2BGR)

        cv2.imshow('Depth', im_d.astype(np.uint8))
        cv2.imshow('Sparse', im_sparse.astype(np.uint8))
        # cv2.imshow('RGB', rgb_im.astype(np.uint8))
        key = cv2.waitKey(10000)
        esc = 27
        if key == esc:
            break


if __name__ == '__main__':
    main()

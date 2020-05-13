
import torch
import torch.nn.functional as F
import cv2
import numpy as np



class Transform:
    def __init__(self, transforms=[]):
        self.transforms=transforms

    def transform(self,input):
        # should be overloaded
        return input

    def __call__(self, input):

        output=self.transform(input)
        for t in self.transforms:
            output=t(output)
        return output

    def append(self, transform):
        self.transforms.append(transform)
        
    def __repr__(self):
        print('transform:')
        for tr in self.transforms:
            assert tr is not None
            print(tr.__class__.__name__)
    def __str__(self):
        return 'Transform'




class Resize(Transform):
    def __init__(self, newsize, mode='bilinear'):
        super().__init__()
        newsizei = [int(newsize[0]), int(newsize[1])]
        assert abs(newsizei[0] - newsize[0]) + abs(newsizei[1] - newsize[1])< 1e-14 , " must be integers"
        self.newsize=newsizei
        self.mode=mode

    def transform(self, image):
        if len(image.shape)==2:
            return F.interpolate(image.unsqueeze(0).unsqueeze(0), size=self.newsize, mode=self.mode,align_corners=False).squeeze(0).squeeze(0)
        if len(image.shape)==3:
            return F.interpolate(image.unsqueeze(0), size=self.newsize, mode=self.mode,align_corners=False).squeeze(0)
        assert False, "bad shape"
    def __str__(self):
        return 'Transform Resize'


class ResizeDisparity(Transform):
    # scales images or scales and adjusts disparites
    def __init__(self, newsize, mode='bilinear'):
        super().__init__()
        newsizei = [int(newsize[0]), int(newsize[1])]
        assert abs(newsizei[0] - newsize[0]) + abs(newsizei[1] - newsize[1]) < 1e-14, " must be integers"
        self.newsize = newsizei
        self.mode = mode

    def transform(self, image):
        assert len(image.shape)==2, " should be for regular tensors of length two (rows,cols)"
        rows,cols = image.shape
        disp=image.unsqueeze(0).unsqueeze(0)


        disp=F.interpolate(disp, size=self.newsize, mode=self.mode,align_corners=False)*torch.tensor(self.newsize[1]/cols)
        disp=disp.squeeze()
        return disp
    def __str__(self):
        return 'Transform ResizeDisparity'



import random



class RandomSwapRGB(Transform):
    # this class makes a single specific indexshuffle
    # this is intended
    def __init__(self):
        super().__init__()
        self.index=[0,1,2]
        random.shuffle(self.index)

    # will be applied to each with the same index swaps!
    def transform(self, image):

        assert(len(image.shape)==3)
        index=self.index
        # save some computation
        if index==[0,1,2]:
            return image

        a = image[index[0], :, :].unsqueeze(0)
        b = image[index[1], :, :].unsqueeze(0)
        c = image[index[2], :, :].unsqueeze(0)

        return torch.cat((a,b,c),0)



class RandomCrop(Transform):
    # this class makes a single specific random crop
    # this is intended
    def __init__(self, input_shape, crop_size=[1216,352]):
        super().__init__()


        rows = input_shape[-2]
        cols = input_shape[-1]
        nrows=crop_size[0]
        ncols=crop_size[1]
        self.r0 = random.randint(0, rows - nrows)
        self.c0 = random.randint(0, cols - ncols)
        self.rend=self.r0+nrows
        self.cend=self.c0+ncols

    def transform(self, image):

        if(len(image.shape)==3):
            return image[:,self.r0:self.rend,self.c0:self.cend]
        if (len(image.shape) == 2):
            return image[self.r0:self.rend, self.c0:self.cend]
        assert False, "only images"
        
        
############# vKITTI Transforms 
        
class Crop(Transform):
    # this class makes a single specific crop
    # this is intended
    def __init__(self, crop_size=[1216,352]):
        super().__init__()
        
        self.nrows=crop_size[0]
        self.ncols=crop_size[1]
        
        

    def transform(self, image):
    
        if(len(image.shape)==3):
            return image[:,0:self.nrows,0:self.ncols]
        if (len(image.shape) == 2):
            return image[0:self.nrows,0:self.ncols]
        assert False, "only images"
        

# the base transform which outputs rgb images in (3,r,c) tensor form
class vkitti_image_transform(Transform):
    def __init__(self):
             super().__init__()
    def transform(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #1242x375
        img = img.astype(np.float32)
        img = img / 255 - 0.5
        img=np.transpose(img, (2,0,1))

        return torch.as_tensor(img)



class vkitti_disparity_transform(Transform):
    def __init__(self):
             super().__init__()
    def transform(self, img):
        # disparity is stored as depth in cm, up to a max of 65535 cm, which is sortof inf...
        img = img.astype(np.float32) # pretty sure they are allready
        # these are guesses
        baseline=0.4 # meters
        focal= 725 # matches kitti
        img = (baseline*focal)/(0.01*img)
        img = np.expand_dims(img, 0)
        return torch.as_tensor(img)

class vkitti_depth_transform(Transform):
    def __init__(self):
             super().__init__()
             
    def transform(self, img):
        # disparity is stored as depth in cm, up to a max of 65535 cm, which is sortof inf...
        img = img.astype(np.float32) # pretty sure they are allready
        # these are guesses
        img = 0.01*img
        img = np.expand_dims(img, 0)
        return torch.as_tensor(img)
    



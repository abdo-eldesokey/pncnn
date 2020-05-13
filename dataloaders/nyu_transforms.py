
import torch
from PIL import Image, ImageEnhance
try:
    import accimage
except ImportError:
    accimage = None

import numbers
import types
import collections

import scipy.ndimage.interpolation as itpl
import scipy.misc as misc

import numpy as np
import cv2


def rgb2grayscale(rgb):
    return rgb[:, :, 0] * 0.2989 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114


class DenseToSparse:
    def __init__(self):
        pass

    def dense_to_sparse(self, rgb, depth):
        pass

    def __repr__(self):
        pass

class UniformSampling(DenseToSparse):
    name = "uar"

    def __init__(self, num_samples, max_depth=np.inf):
        DenseToSparse.__init__(self)
        self.num_samples = num_samples
        self.max_depth = max_depth

    def __repr__(self):
        return "%s{ns=%d,md=%f}" % (self.name, self.num_samples, self.max_depth)

    def dense_to_sparse(self, rgb, depth):
        """
        Samples pixels with `num_samples`/#pixels probability in `depth`.
        Only pixels with a maximum depth of `max_depth` are considered.
        If no `max_depth` is given, samples in all pixels
        """
        mask_keep = depth > 0
        if self.max_depth is not np.inf:
            mask_keep = np.bitwise_and(mask_keep, depth <= self.max_depth)
        n_keep = np.count_nonzero(mask_keep)
        if n_keep == 0:
            return mask_keep
        else:
            prob = float(self.num_samples) / n_keep
            return np.bitwise_and(mask_keep, np.random.uniform(0, 1, depth.shape) < prob)


class SimulatedStereo(DenseToSparse):
    name = "sim_stereo"

    def __init__(self, num_samples, max_depth=np.inf, dilate_kernel=3, dilate_iterations=1):
        DenseToSparse.__init__(self)
        self.num_samples = num_samples
        self.max_depth = max_depth
        self.dilate_kernel = dilate_kernel
        self.dilate_iterations = dilate_iterations

    def __repr__(self):
        return "%s{ns=%d,md=%f,dil=%d.%d}" % \
               (self.name, self.num_samples, self.max_depth, self.dilate_kernel, self.dilate_iterations)

    # We do not use cv2.Canny, since that applies non max suppression
    # So we simply do
    # RGB to intensitities
    # Smooth with gaussian
    # Take simple sobel gradients
    # Threshold the edge gradient
    # Dilatate
    def dense_to_sparse(self, rgb, depth):
        gray = rgb2grayscale(rgb)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
        gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)

        depth_mask = np.bitwise_and(depth != 0.0, depth <= self.max_depth)

        edge_fraction = float(self.num_samples) / np.size(depth)

        mag = cv2.magnitude(gx, gy)
        min_mag = np.percentile(mag[depth_mask], 100 * (1.0 - edge_fraction))
        mag_mask = mag >= min_mag

        if self.dilate_iterations >= 0:
            kernel = np.ones((self.dilate_kernel, self.dilate_kernel), dtype=np.uint8)
            cv2.dilate(mag_mask.astype(np.uint8), kernel, iterations=self.dilate_iterations)

        mask = np.bitwise_and(mag_mask, depth_mask)
        return mask


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def adjust_brightness(img, brightness_factor):
    """Adjust brightness of an Image.

    Args:
        img (PIL Image): PIL Image to be adjusted.
        brightness_factor (float):  How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.

    Returns:
        PIL Image: Brightness adjusted image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)
    return img


def adjust_contrast(img, contrast_factor):
    """Adjust contrast of an Image.

    Args:
        img (PIL Image): PIL Image to be adjusted.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.

    Returns:
        PIL Image: Contrast adjusted image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    return img


def adjust_saturation(img, saturation_factor):
    """Adjust color saturation of an image.

    Args:
        img (PIL Image): PIL Image to be adjusted.
        saturation_factor (float):  How much to adjust the saturation. 0 will
            give a black and white image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.

    Returns:
        PIL Image: Saturation adjusted image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)
    return img


def adjust_hue(img, hue_factor):
    """Adjust hue of an image.

    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.

    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.

    See https://en.wikipedia.org/wiki/Hue for more details on Hue.

    Args:
        img (PIL Image): PIL Image to be adjusted.
        hue_factor (float):  How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.

    Returns:
        PIL Image: Hue adjusted image.
    """
    if not(-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor is not in [-0.5, 0.5].'.format(hue_factor))

    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    input_mode = img.mode
    if input_mode in {'L', '1', 'I', 'F'}:
        return img

    h, s, v = img.convert('HSV').split()

    np_h = np.array(h, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over='ignore'):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, 'L')

    img = Image.merge('HSV', (h, s, v)).convert(input_mode)
    return img


def adjust_gamma(img, gamma, gain=1):
    """Perform gamma correction on an image.

    Also known as Power Law Transform. Intensities in RGB mode are adjusted
    based on the following equation:

        I_out = 255 * gain * ((I_in / 255) ** gamma)

    See https://en.wikipedia.org/wiki/Gamma_correction for more details.

    Args:
        img (PIL Image): PIL Image to be adjusted.
        gamma (float): Non negative real number. gamma larger than 1 make the
            shadows darker, while gamma smaller than 1 make dark regions
            lighter.
        gain (float): The constant multiplier.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    if gamma < 0:
        raise ValueError('Gamma should be a non-negative real number')

    input_mode = img.mode
    img = img.convert('RGB')

    np_img = np.array(img, dtype=np.float32)
    np_img = 255 * gain * ((np_img / 255) ** gamma)
    np_img = np.uint8(np.clip(np_img, 0, 255))

    img = Image.fromarray(np_img, 'RGB').convert(input_mode)
    return img


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class ToTensor(object):
    """Convert a ``numpy.ndarray`` to tensor.

    Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W).
    """

    def __call__(self, img):
        """Convert a ``numpy.ndarray`` to tensor.

        Args:
            img (numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        if not(_is_numpy_image(img)):
            raise TypeError('img should be ndarray. Got {}'.format(type(img)))

        if isinstance(img, np.ndarray):
            # handle numpy array
            if img.ndim == 3:
                img = torch.from_numpy(img.transpose((2, 0, 1)).copy())
            elif img.ndim == 2:
                img = torch.from_numpy(img.copy())
            else:
                raise RuntimeError('img should be ndarray with 2 or 3 dimensions. Got {}'.format(img.ndim))

            # backward compatibility
            # return img.float().div(255)
            return img.float()


class NormalizeNumpyArray(object):
    """Normalize a ``numpy.ndarray`` with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(M1,..,Mn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``numpy.ndarray`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray): Image of size (H, W, C) to be normalized.

        Returns:
            Tensor: Normalized image.
        """
        if not(_is_numpy_image(img)):
            raise TypeError('img should be ndarray. Got {}'.format(type(img)))
        # TODO: make efficient
        print(img.shape)
        for i in range(3):
            img[:,:,i] = (img[:,:,i] - self.mean[i]) / self.std[i]
        return img

class NormalizeTensor(object):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(M1,..,Mn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        if not _is_tensor_image(tensor):
            raise TypeError('tensor is not a torch image.')
        # TODO: make efficient
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor

class Rotate(object):
    """Rotates the given ``numpy.ndarray``.

    Args:
        angle (float): The rotation angle in degrees.
    """

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray (C x H x W)): Image to be rotated.

        Returns:
            img (numpy.ndarray (C x H x W)): Rotated image.
        """

        # order=0 means nearest-neighbor type interpolation
        return itpl.rotate(img, self.angle, reshape=False, prefilter=False, order=0)


class Resize(object):
    """Resize the the given ``numpy.ndarray`` to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation='nearest'):
        assert isinstance(size, int) or isinstance(size, float) or \
               (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.
        Returns:
            PIL Image: Rescaled image.
        """
        if img.ndim == 3:
            return misc.imresize(img, self.size, self.interpolation)
        elif img.ndim == 2:
            return misc.imresize(img, self.size, self.interpolation, 'F')
        else:
            RuntimeError('img should be ndarray with 2 or 3 dimensions. Got {}'.format(img.ndim))


class CenterCrop(object):
    """Crops the given ``numpy.ndarray`` at the center.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for center crop.

        Args:
            img (numpy.ndarray (C x H x W)): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for center crop.
        """
        h = img.shape[0]
        w = img.shape[1]
        th, tw = output_size
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))

        # # randomized cropping
        # i = np.random.randint(i-3, i+4)
        # j = np.random.randint(j-3, j+4)

        return i, j, th, tw

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray (C x H x W)): Image to be cropped.

        Returns:
            img (numpy.ndarray (C x H x W)): Cropped image.
        """
        i, j, h, w = self.get_params(img, self.size)

        """
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
        """
        if not(_is_numpy_image(img)):
            raise TypeError('img should be ndarray. Got {}'.format(type(img)))
        if img.ndim == 3:
            return img[i:i+h, j:j+w, :]
        elif img.ndim == 2:
            return img[i:i + h, j:j + w]
        else:
            raise RuntimeError('img should be ndarray with 2 or 3 dimensions. Got {}'.format(img.ndim))


class Lambda(object):
    """Apply a user-defined lambda as a transform.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)


class HorizontalFlip(object):
    """Horizontally flip the given ``numpy.ndarray``.

    Args:
        do_flip (boolean): whether or not do horizontal flip.

    """

    def __init__(self, do_flip):
        self.do_flip = do_flip

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray (C x H x W)): Image to be flipped.

        Returns:
            img (numpy.ndarray (C x H x W)): flipped image.
        """
        if not(_is_numpy_image(img)):
            raise TypeError('img should be ndarray. Got {}'.format(type(img)))

        if self.do_flip:
            return np.fliplr(img)
        else:
            return img


class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness (float): How much to jitter brightness. brightness_factor
            is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
        contrast (float): How much to jitter contrast. contrast_factor
            is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
        saturation (float): How much to jitter saturation. saturation_factor
            is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
        hue(float): How much to jitter hue. hue_factor is chosen uniformly from
            [-hue, hue]. Should be >=0 and <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []
        if brightness > 0:
            brightness_factor = np.random.uniform(max(0, 1 - brightness), 1 + brightness)
            transforms.append(Lambda(lambda img: adjust_brightness(img, brightness_factor)))

        if contrast > 0:
            contrast_factor = np.random.uniform(max(0, 1 - contrast), 1 + contrast)
            transforms.append(Lambda(lambda img: adjust_contrast(img, contrast_factor)))

        if saturation > 0:
            saturation_factor = np.random.uniform(max(0, 1 - saturation), 1 + saturation)
            transforms.append(Lambda(lambda img: adjust_saturation(img, saturation_factor)))

        if hue > 0:
            hue_factor = np.random.uniform(-hue, hue)
            transforms.append(Lambda(lambda img: adjust_hue(img, hue_factor)))

        np.random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray (C x H x W)): Input image.

        Returns:
            img (numpy.ndarray (C x H x W)): Color jittered image.
        """
        if not(_is_numpy_image(img)):
            raise TypeError('img should be ndarray. Got {}'.format(type(img)))

        pil = Image.fromarray(img)
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        return np.array(transform(pil))

class Crop(object):
    """Crops the given PIL Image to a rectangular region based on a given
    4-tuple defining the left, upper pixel coordinated, hight and width size.

    Args:
        a tuple: (upper pixel coordinate, left pixel coordinate, hight, width)-tuple
    """

    def __init__(self, i, j, h, w):
        """
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
        """
        self.i = i
        self.j = j
        self.h = h
        self.w = w

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray (C x H x W)): Image to be cropped.
        Returns:
            img (numpy.ndarray (C x H x W)): Cropped image.
        """

        i, j, h, w = self.i, self.j, self.h, self.w

        if not(_is_numpy_image(img)):
            raise TypeError('img should be ndarray. Got {}'.format(type(img)))
        if img.ndim == 3:
            return img[i:i + h, j:j + w, :]
        elif img.ndim == 2:
            return img[i:i + h, j:j + w]
        else:
            raise RuntimeError(
                'img should be ndarray with 2 or 3 dimensions. Got {}'.format(img.ndim))

    def __repr__(self):
        return self.__class__.__name__ + '(i={0},j={1},h={2},w={3})'.format(
            self.i, self.j, self.h, self.w)

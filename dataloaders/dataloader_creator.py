import os
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader

# Kitti_dept
from dataloaders.KittiDepthDataset import KittiDepthDataset

# vKitti
from .vKittiDataset import VKittiDataset, VkittiDatasetWrapper
from torch.utils.data import RandomSampler

# NYU
from dataloaders.nyu_transforms import UniformSampling, SimulatedStereo


def create_dataloader(args, eval_mode=False):
    print('==> Loading dataset "{}" .. \n'.format(args.dataset))
    if args.dataset_path == 'machine':
        get_data_set_path(args)

    if args.dataset == 'kitti_depth':
        train_loader, val_loader = create_kitti_depth_dataloader(args, eval_mode)
        val_set = args.val_ds
    elif args.dataset == 'vkitti':
        train_loader, val_loader = create_vkitti_depth_dataloader(args, eval_mode)
        val_set = 'val'
    elif args.dataset == 'nyudepthv2' or args.dataset == 'kitti_odo':
        train_loader, val_loader = create_nyu_odo_dataloader(args, eval_mode)
        val_set = 'val'

    if not eval_mode:
        print('- Found {} images in "{}" folder.\n'.format(len(train_loader.dataset), 'train'))

    print('- Found {} images in "{}" folder.\n'.format(len(val_loader.dataset), val_set))
    print('==> Dataset "{}" was loaded successfully!'.format(args.dataset))

    return train_loader, val_loader


################### KITTI DEPTH ###################
def create_kitti_depth_dataloader(args, eval_mode=False):
    # Input images are 16-bit, but only 15-bits are utilized, so we normalized the data to [0:1] using a normalization factor
    norm_factor = args.norm_factor
    invert_depth = args.train_disp
    ds_dir = args.dataset_path
    rgb_dir = args.raw_kitti_path
    train_on = args.train_on
    rgb2gray = args.rgb2gray
    val_set = args.val_ds
    data_aug = args.data_aug if hasattr(args, 'data_aug') else False

    if args.modality == 'rgbd':
        load_rgb = True
    else:
        load_rgb = False

    train_loader = []
    val_loader = []

    if eval_mode is not True:
        ###### Training Set ######
        trans_list = [transforms.CenterCrop((352, 1216))]
        train_transform = transforms.Compose(trans_list)
        train_dataset = KittiDepthDataset(ds_dir, setname='train', transform=train_transform,
                                          norm_factor=norm_factor, invert_depth=invert_depth,
                                          load_rgb=load_rgb, kitti_rgb_path=rgb_dir, rgb2gray=rgb2gray, hflip=data_aug)

        # Select the desired number of images from the training set
        if train_on != 'full':
            import random
            training_idxs = np.array(random.sample(range(0, len(train_dataset)), int(train_on)))
            train_dataset.depth = train_dataset.depth[training_idxs]
            train_dataset.gt = train_dataset.gt[training_idxs]

        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.workers)

    # Validation set
    if val_set == 'val':
        ###### Validation Set ######
        val_dataset = KittiDepthDataset(ds_dir, setname='val', transform=None,
                                        norm_factor=norm_factor, invert_depth=invert_depth,
                                        load_rgb=load_rgb, kitti_rgb_path=rgb_dir, rgb2gray=rgb2gray)

        val_loader = DataLoader(val_dataset, shuffle=False, batch_size=1, num_workers=args.workers)

    elif val_set == 'selval':
        ###### Selected Validation set ######
        val_dataset = KittiDepthDataset(ds_dir, setname='selval', transform=None,
                                        norm_factor=norm_factor, invert_depth=invert_depth,
                                        load_rgb=load_rgb, kitti_rgb_path=rgb_dir, rgb2gray=rgb2gray)

        val_loader = DataLoader(val_dataset, shuffle=False, batch_size=1, num_workers=args.workers)

    elif val_set == 'test':
        ###### Test set ######
        val_dataset = KittiDepthDataset(ds_dir, setname='test', transform=None,
                                        norm_factor=norm_factor, invert_depth=invert_depth,
                                        load_rgb=load_rgb, kitti_rgb_path=rgb_dir, rgb2gray=rgb2gray)

        val_loader = DataLoader(val_dataset, shuffle=False, batch_size=1, num_workers=args.workers)

    return train_loader, val_loader


################### vKITTI DEPTH ###################
def create_vkitti_depth_dataloader(args, eval_mode=False):
    path = args.dataset_path
    rgb_dir = args.raw_kitti_path
    train_on = args.train_on

    if args.modality == 'rgbd':
        load_rgb = True
    else:
        load_rgb = False

    vkd = VKittiDataset(path, args)
    vkd.init()

    train, val, test = vkd.get_training_validation_testing()

    # Number of training images
    if args.train_on != 'full':
        train = train[0:int(train_on)]

    train_loader = None
    val_loader = None
    if eval_mode is not True:
        train = VkittiDatasetWrapper(train, load_rgb)
        sampler = RandomSampler(train)
        train_loader = DataLoader(train, num_workers=args.workers, batch_size=args.batch_size, sampler=sampler,
                                  drop_last=True)

    if args.val_ds == 'val':
        val = VkittiDatasetWrapper(val, load_rgb)
        val_loader = DataLoader(val, num_workers=args.workers, batch_size=1, drop_last=False)
    elif args.val_ds == 'test':
        val = VkittiDatasetWrapper(test, load_rgb)
        val_loader = DataLoader(test, num_workers=args.workers, batch_size=1, drop_last=False)

    dataset_sizes = {'train': len(train), 'val': len(val)}

    print(dataset_sizes)

    return train_loader, val_loader


################### NYUDEPTHv2 & KITTI-ODO###################
def create_nyu_odo_dataloader(args, eval_mode=False):
    # Data loading code
    path = args.dataset_path
    traindir = os.path.join(path, 'train')
    valdir = os.path.join(path, 'val')

    train_loader = None
    val_loader = None

    shift = args.shift if hasattr(args, 'shift') else None
    rotate = args.rotate if hasattr(args, 'rotate') else None

    # sparsifier is a class for generating random sparse depth input from the ground truth
    sparsifier = None
    max_depth = args.max_depth if args.max_depth >= 0.0 else np.inf
    if args.sparsifier == UniformSampling.name:
        sparsifier = UniformSampling(num_samples=args.num_samples, max_depth=max_depth)
    elif args.sparsifier == SimulatedStereo.name:
        sparsifier = SimulatedStereo(num_samples=args.num_samples, max_depth=max_depth)

    if args.dataset == 'nyudepthv2':  ###### NYU ########
        from dataloaders.NYUDataset import NYUDataset

        if not eval_mode:
            train_dataset = NYUDataset(traindir, type='train', modality=args.modality, sparsifier=sparsifier,
                                       shift=shift, rotate=rotate)

        val_dataset = NYUDataset(valdir, type='val', modality=args.modality, sparsifier=sparsifier,
                                 shift=shift, rotate=rotate)

    elif args.dataset == 'kitti_odo':  ###### KITTI-ODO ########
        from dataloaders.KittiOdoDataset import KittiOdoDataset

        if not eval_mode:
            train_dataset = KittiOdoDataset(traindir, type='train', modality=args.modality, sparsifier=sparsifier)

        val_dataset = KittiOdoDataset(valdir, type='val', modality=args.modality, sparsifier=sparsifier)

    # Select the desired number of images from the training set
    if args.train_on != 'full':
        train_dataset.imgs = train_dataset.imgs[0:int(args.train_on)]

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.workers,
                            pin_memory=True)

    # put construction of train loader here, for those who are interested in testing only
    if not eval_mode:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                  pin_memory=True, sampler=None, worker_init_fn=lambda work_id: np.random.seed(work_id))

    return train_loader, val_loader


def get_data_set_path(args):
    import platform
    machine = platform.node()
    if args.dataset == 'kitti_depth':
        if machine == 'nikola':
            args.dataset_path = '/media/DataDrive2/Abdo/datasets/kitti_depth/'
            args.raw_kitti_path = '/media/DataDrive2/Abdo/datasets/kitti_rgb/'
        elif machine == 'abdel62-PC':
            args.dataset_path = '/ssd/datasets/kitti_depth'
            args.raw_kitti_path = '/ssd/datasets/kitti_raw'
    elif args.dataset == 'nyudepthv2':
        if machine == 'abdel62-PC':
            args.dataset_path = '/ssd/datasets/nyu-depth-v2/nyudepthv2'


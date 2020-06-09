# Probabilistic Normalized Convolutional Neural Networks (pNCNN)

This is the official PyTorch implementation for "Uncertainty-Aware CNNs for Depth Completion: Uncertainty from Beginning to End" presented at CVPR 2020, Seattle, USA.

[[PDF]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Eldesokey_Uncertainty-Aware_CNNs_for_Depth_Completion_Uncertainty_from_Beginning_to_End_CVPR_2020_paper.pdf) [[ArXiv (With Supplementary)]](https://arxiv.org/abs/2006.03349) [[1min Video]](https://www.youtube.com/watch?v=Iw_yk-UoKEo&feature=youtu.be) [[Slides]]

<p align="center">
  <img src="imgs/teaser2.gif"/>
</p>

```
@InProceedings{Eldesokey_2020_CVPR,
author = {Eldesokey, Abdelrahman and Felsberg, Michael and Holmquist, Karl and Persson, Michael},
title = {Uncertainty-Aware CNNs for Depth Completion: Uncertainty from Beginning to End},
booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```
---

## Dependecies
The code was developed using Python 3.7.4 and PyTorch 1.4, but it should work on any PyTorch version > 1.1

* pytorch>1.1
* torchvision>0.5.0
* json
* matplotlib
* opencv
* h5py
 
---

## Datasets

### Kitti-Depth
To download the Kitti-Depth dataset, use the provided Python script [`dataloaders/download_kitti_depth_rgb.py`](dataloaders/download_kitti_depth_rgb.py). 

*Remeber to edit the script first to set download directories.*

### NYU-Depth-v2
Download and extract the dataset in h5 format provided from [sparse-to-dense](https://github.com/fangchangma/sparse-to-dense.pytorch#requirements).

```
wget http://datasets.lids.mit.edu/sparse-to-dense/data/nyudepthv2.tar.gz
tar -xvf nyudepthv2.tar.gz && rm -f nyudepthv2.tar.gz
```

---

## Training

Experiments are stored in `workspace` directory, where you can have different workspaces in sub-directories. 

To train a new experiment, you should create a new directory with the name of the experiment inside your workspace directory which has the following files:
* `network.py` which has the desired architecture.
* `args.json` which has the experiment arguments. 

You can copy these files from any of the pretrained models and modify it.

To run the training, you need to run the following command:
```bash
python main.py --ws <WORKSPACE> --exp <EXP> --args <ARGS>
```

`--ws` is the name of the sub-direcotry inside `workspace` that has your experiment.

`--exp` is the name of the experiment.

`--args` You have two options: either to set this argument to `json` which will load all arguments from `args.json` described above, 
*OR* discard it and set all the arguments in the terminal manually.

  
### Example 
To create an experiemnt called `my_experiment` inside a workspace called `my_workspace`, then you should create a directory for the experiments at `workspace/my_workspace/my_experiment`. This direcotry should have two main files `network.py` and `args.json` as described above. 

You need to modify the following arguments in the json file to match your new experiments: [`exp`, `workspace`, `dataset`, `dataset_path`]. Other arguments, you can change as needed.

Now you are ready to start training by calling:
```bash
python main.py --ws my_workspace --exp my_experiment --args json
```

---

## Logging 
Tensorboard is supported by default and you can initiate it as usual by calling:
```bash
tensorboard --logdir=workspace/my_workspace
```
By default, tensorboard log files are save to the directory `tb_log` inside the experiment directory.

Also, the evaluation metrics are saved after each epoch both for the training and the test set as CSV files inside the experiment directory.

---

## Pretrained Models
We provide the pretrained models for the `KITTI-Depth` dataset and the `NYU-Depth-v2` dataset inside `workspace/kitti` and `workspace/nyu` respectively.

---

## Resuming Training
To resume training, you can call:
```bash
python main.py --resume <path-to-checkpoint>
```
By default, the argument will be loaded from the checkpoint. If you want to change some arguments, you can edit `args.json` for the experiment and it will override the arguments in the checkpoints.

---

## Testing
To test a pretrained model, you can call:
```bash
python main.py --evaluate <path-to-checkpoint>
```

---

## Remarks
If you use our code or our paper, please consider citing us. The bibtex is provided above.

If you have questions, please create an issue.




import numpy as np
import torch
import matplotlib.pylab as plt
import cv2

ratio_removed = np.linspace(0, 1, 100, endpoint=False)


def sparsification_plot(var_vec, err_vec, uncert_type='c'):
    # Sort the error
    #print('Sorting Error ...')
    err_vec_sorted, _ = torch.sort(err_vec)
    #print(' Done!')

    # Calculate the error when removing a fraction pixels with error
    n_valid_pixels = len(err_vec)
    rmse_err = []
    for i, r in enumerate(ratio_removed):
        mse_err_slice = err_vec_sorted[0:int((1-r)*n_valid_pixels)]
        rmse_err.append(torch.sqrt(mse_err_slice.mean()).cpu().numpy())

    # Normalize RMSE
    rmse_err = rmse_err / rmse_err[0]

    ###########################################

    # Sort by variance
    #print('Sorting Variance ...')
    if uncert_type == 'c':
        var_vec = torch.sqrt(var_vec)
        _, var_vec_sorted_idxs = torch.sort(var_vec, descending=True)
    elif uncert_type == 'v':
        #var_vec = torch.exp(var_vec)
        var_vec = torch.sqrt(var_vec)
        _, var_vec_sorted_idxs = torch.sort(var_vec, descending=False)
    #print(' Done!')

    # Sort error by variance
    err_vec_sorted_by_var = err_vec[var_vec_sorted_idxs]

    rmse_err_by_var = []
    for i, r in enumerate(ratio_removed):
        mse_err_slice = err_vec_sorted_by_var[0:int((1 - r) * n_valid_pixels)]
        rmse_err_by_var.append(torch.sqrt(mse_err_slice.mean()).cpu().numpy())

    # Normalize RMSE
    rmse_err_by_var = rmse_err_by_var / max(rmse_err_by_var)

    #plt.plot(ratio_removed, rmse_err, '--')
    #plt.plot(ratio_removed, rmse_err_by_var, '-r')
    #plt.show()
    return rmse_err, rmse_err_by_var


if __name__ == '__main__':
    from pathlib import Path
    import glob
    ws = 'bdl_nikola'
    expr = 'l2_var_est_ens'
    epoch = 6
    if ws=='bdl':
        pred_path = '../workspace/' + ws + '/' + expr + '/epoch_' + str(
            epoch) + '_kitti_depth_selval_output'
        var_path = '../workspace/' + ws + '/' + expr + '/epoch_' + str(
            epoch) + '_kitti_depth_selval_cout'
        target_path = '/ssd/datasets/KITTI_Depth/val_selection_cropped/groundtruth_depth'
    elif ws == 'bdl_nikola':
        pred_path = '../workspace/' + ws + '/' + expr + '/epoch_' + str(
            epoch) + '_kitti_depth_selval_output'
        var_path = '../workspace/' + ws + '/' + expr + '/epoch_' + str(
            epoch) + '_kitti_depth_selval_cout'
        target_path = '/media/DataDrive2/Abdo/datasets/kitti_depth/depth_selection/val_selection_cropped/groundtruth_depth'
    elif ws == 'bdl_nyu':
        pred_path = '../workspace/' + ws + '/' + expr + '/epoch_' + str(
            epoch) + '_nyudepthv2_selval_output'
        var_path = '../workspace/' + ws + '/' + expr + '/epoch_' + str(
            epoch) + '_nyudepthv2_selval_cout'
        target_path = '/ssd/datasets/nyu-depth-v2/nyu_val_gt'

    pred_dir = sorted(glob.glob(pred_path + '/*.png'))
    target_dir = sorted(glob.glob(target_path + '/*.png'))
    var_dir = sorted(glob.glob(var_path + '/*.png'))

    num_image = len(pred_dir)

    rmses_by_err = np.zeros(100)
    rmses_by_var = np.zeros(100)
    for i in range(num_image):
        print('Processing image {}'.format(i))
        pred = cv2.imread(pred_dir[i], -1)
        target = cv2.imread(target_dir[i], -1)
        var = cv2.imread(var_dir[i], -1)

        rmse_by_err, rmse_by_var = calculate_ause(pred, target, var, inv_var=True)

        rmses_by_err += rmse_by_err
        rmses_by_var += rmse_by_var

    rmses_by_err /= num_image
    rmses_by_var /= num_image

    ause = np.trapz(rmses_by_var-rmses_by_err, ratio_removed)
    print(ause)

    plt.plot(ratio_removed, rmses_by_err, '--')
    plt.plot(ratio_removed, rmses_by_var, '-r')
    plt.show()
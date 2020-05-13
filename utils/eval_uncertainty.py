import torch
import numpy as np
import argparse
import os
import sys
from dataloaders.dataloader_creator import create_dataloader
from utils.sparsification_plot import sparsification_plot
import matplotlib.pylab as plt


def eval_ause(model, dataloader, args, epoch, uncertainty_comp='a', show_plot=False, uncert_type='c', from_main=False):

    print('\n==> Evaluating Uncertainty for Epoch [{}]:'.format(epoch))

    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")

    model.eval()  # Swith to evaluate mode

    # Save output images
    num_image = len(dataloader.dataset)
    ratio_removed = np.linspace(0, 1, 100, endpoint=False)
    all_err = torch.tensor([]).to(device)
    all_var = torch.tensor([]).to(device)
    with torch.no_grad():  # Disable gradients computations
        for i, (input, target) in enumerate(dataloader):

            input, target = input.to(device), target.to(device)

            torch.cuda.synchronize()  # Wait for all kernels to finish

            out = model(input)

            if uncertainty_comp == 'a+e':
                # Epistimic
                eps = 1e-16
                c1 = 1/(model.cout1+eps)#1 - model.cout1
                c2 = 1/(model.cout2+eps)#1 - model.cout2
                c3 = 1/(model.cout3+eps)#1 - model.cout3
                c4 = 1/(model.cout4+eps)#1 - model.cout4

                x1 = model.xout1 / 70
                x2 = model.xout2 / 70
                x3 = model.xout3 / 70
                x4 = model.xout4 / 70

                mu = torch.mean(torch.cat((x1, x2, x3, x4), 1), 1).unsqueeze(1)

                s2 = torch.mean(
                    torch.cat(((x1 - mu) ** 2 + c1, (x2 - mu) ** 2 + c2, (x3 - mu) ** 2 + c3, (x4 - mu) ** 2 + c4), 1),
                    1).unsqueeze(1)

            for j in range(input.shape[0]):
                #print('Processing image {}'.format(i * input.shape[0] + j))
                t = target[j, 0, :, :]
                pred = out[j, 0, :, :]

                if uncertainty_comp == 'a+e':
                    # pred = mu[j,0,:,:].cpu().numpy()
                    var_alea = 1/(out[j, 1, :, :]+eps) #1 - out[j, 1, :, :]
                    var_epi = s2[j, 0, :, :]
                    var = var_epi
                else:
                    var_alea = out[j, 1, :, :]
                    var = var_alea

                valid = (t != 0)
                target_valid = t[valid]
                pred_valid = pred[valid]
                var_valid = var[valid]
                err_valid = (target_valid - pred_valid) ** 2

                #all_var = np.concatenate((all_var, var_valid.cpu().numpy().astype(np.double)))
                #all_err = np.concatenate((all_err, err_valid.cpu().numpy().astype(np.double)))
                all_var = torch.cat((all_var, var_valid), 0)
                all_err = torch.cat((all_err, err_valid), 0)

    rmse_by_err, rmse_by_var = sparsification_plot(all_var, all_err, uncert_type=uncert_type)
    ause = np.trapz(rmse_by_var - rmse_by_err, ratio_removed)
    print('- AUSE metric is {:.5f}.'.format(ause))

    plt.clf()
    plt.xlabel('Perc. of removed pixels.')
    plt.ylabel('RMSE (normalized)')
    plt.plot(ratio_removed, rmse_by_err, '--')
    plt.plot(ratio_removed, rmse_by_var, '-r')
    plt.legend(['Oracle', 'CNN Pred.'], loc='lower left')
    plt.text(0.6, 0.95, "AUSE[{}]={:.5f}".format(uncertainty_comp, ause), fontsize=14)

    if from_main:
        ws_path = os.path.join('..', 'workspace/', args.workspace)
    else:
        ws_path = os.path.join('workspace/', args.workspace)
    exp = args.exp
    exp_dir = os.path.join(ws_path, exp)

    #try:
    #    fname = 'checkpoint_{}_spep.png'.format(str(epoch))
    #    plt.savefig(os.path.join(exp_dir, fname), bbox_inches='tight')
    #except:
    #    print("- Could not save the sparsification plot at {}".format(os.path.join(exp_dir, fname)))

    if show_plot:
        plt.show()
    return ause, plt.gcf()


if __name__ == '__main__':

    # Construct the parser
    parser = argparse.ArgumentParser(description='Uncertainty Evaluation ')

    # Mode selection
    parser.add_argument('-c', type=str, help='The path to the checkpoint.')

    parser.add_argument('-t', type=str, default='a',
                        help='Uncertainty Type ("a":aleatoric, "e":epistimic, "a+e" both')

    parser.add_argument('-d', type=str, default='c',
                        help='Uncertainty Type ("c":confidence, "v":variance')

    args = parser.parse_args()

    # Define paths
    uncertainty_comp = args.t
    uncert_type = args.d
    chkpt_path = args.c
    exp_dir = os.path.dirname(os.path.abspath(chkpt_path))  # Experiment directory
    sys.path.append(exp_dir)

    # Check that the checkpoint file exist
    assert os.path.isfile(chkpt_path), "- No checkpoint found at: {}".format(chkpt_path)
    # Load checkpoint
    print('- Loading checkpoint:', chkpt_path)

    # Load the checkpoint
    checkpoint = torch.load(chkpt_path)

    # Assign some local variables
    args = checkpoint['args']
    start_epoch = checkpoint['epoch']
    best_result = checkpoint['best_result']
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    model = checkpoint['model'].to(device)
    print('- Checkpoint {} was loader successfully.:'.format(chkpt_path))
    args.val_ds = 'selval'

    _, val_loader = create_dataloader(args, eval_mode=True)

    eval_ause(model, val_loader, args, epoch=start_epoch, uncertainty_comp=uncertainty_comp, show_plot=True, uncert_type=uncert_type, from_main=True)







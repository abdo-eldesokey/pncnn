#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 14:16:29 2019

@author: abdel62
"""
import os
import sys
import importlib
import time
import datetime

import torch
from torch.optim import SGD, Adam
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from utils.args_parser import args_parser, save_args, print_args, initialize_args,  compare_args_w_json
from dataloaders.dataloader_creator import create_dataloader
from utils.error_metrics import AverageMeter, create_error_metric, LogFile
from utils.save_output_images import create_out_image_saver, colored_depthmap_tensor
from utils.checkpoints import save_checkpoint
from common.losses import get_loss_fn
from utils.eval_uncertainty import eval_ause


def main():
    # Make some variable global
    global args, train_csv, test_csv, exp_dir, best_result, device, tb_writer, tb_freq

    # Args parser 
    args = args_parser()

    start_epoch = 0
############ EVALUATE MODE ############
    if args.evaluate:  # Evaluate mode
        print('\n==> Evaluation mode!')

        # Define paths
        chkpt_path = args.evaluate

        # Check that the checkpoint file exist
        assert os.path.isfile(chkpt_path), "- No checkpoint found at: {}".format(chkpt_path)

        # Experiment director
        exp_dir = os.path.dirname(os.path.abspath(chkpt_path))
        sys.path.append(exp_dir)

        # Load checkpoint
        print('- Loading checkpoint:', chkpt_path)

        # Load the checkpoint
        checkpoint = torch.load(chkpt_path)

        # Assign some local variables
        args = checkpoint['args']
        start_epoch = checkpoint['epoch']
        best_result = checkpoint['best_result']
        print('- Checkpoint was loaded successfully.')

        # Compare the checkpoint args with the json file in case I wanted to change some args
        compare_args_w_json(args, exp_dir, start_epoch+1)
        args.evaluate = chkpt_path

        device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
        model = checkpoint['model'].to(device)

        print_args(args)

        _, val_loader = create_dataloader(args, eval_mode=True)

        loss = get_loss_fn(args).to(device)

        evaluate_epoch(val_loader, model, loss, start_epoch)

        return  # End program

############ RESUME MODE ############
    elif args.resume:  # Resume mode
        print('\n==> Resume mode!')

        # Define paths
        chkpt_path = args.resume
        assert os.path.isfile(chkpt_path), "- No checkpoint found at: {}".format(chkpt_path)

        # Experiment directory
        exp_dir = os.path.dirname(os.path.abspath(chkpt_path))
        sys.path.append(exp_dir)

        # Load checkpoint
        print('- Loading checkpoint:', chkpt_path)
        checkpoint = torch.load(chkpt_path)
        args = checkpoint['args']
        start_epoch = checkpoint['epoch'] + 1
        best_result = checkpoint['best_result']
        print('- Checkpoint ({}) was loaded successfully!\n'.format(checkpoint['epoch']))

        # Compare the checkpoint args with the json file in case I wanted to change some args
        compare_args_w_json(args, exp_dir, start_epoch)
        args.resume = chkpt_path

        device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
        model = checkpoint['model'].to(device)
        optimizer = checkpoint['optimizer']

        print_args(args)

        train_loader, val_loader = create_dataloader(args, eval_mode=False)

############ NEW EXP MODE ############
    else:  # New Exp
        print('\n==> Starting a new experiment "{}" \n'.format(args.exp))

        # Check if experiment exists
        ws_path = os.path.join('workspace/', args.workspace)
        exp = args.exp
        exp_dir = os.path.join(ws_path, exp)
        assert os.path.isdir(exp_dir), '- Experiment "{}" not found!'.format(exp)

        # Which device to use
        device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")

        # Add the experiment's folder to python path
        sys.path.append(exp_dir)

        print_args(args)

        # Create dataloader
        train_loader, val_loader = create_dataloader(args, eval_mode=False)

        # import the model
        f = importlib.import_module('network')
        model = f.CNN().to(device)
        print('\n==> Model "{}" was loaded successfully!'.format(model.__name__))

        # Optimize only parameters that requires_grad
        parameters = filter(lambda p: p.requires_grad, model.parameters())

        # Create Optimizer
        if args.optimizer.lower() == 'sgd':
            optimizer = SGD(parameters, lr=args.lr, momentum=args.momentum,
                            weight_decay=args.weight_decay)
        elif args.optimizer.lower() == 'adam':
            optimizer = Adam(parameters, lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)

############ IF RESUME/NEW EXP ############
    # Error metrics that are set to the worst
    best_result = create_error_metric(args)
    best_result.set_to_worst()

    # Tensorboard
    tb = args.tb_log if hasattr(args, 'tb_log') else False
    tb_freq = args.tb_freq if hasattr(args, 'tb_freq') else 1000
    tb_writer = None
    if tb:
        tb_writer = SummaryWriter(os.path.join(exp_dir, 'tb_log', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))

    # Create Loss
    loss = get_loss_fn(args).to(device)

    # Define Learning rate decay
    lr_decayer = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_factor, last_epoch=start_epoch-1)

    # Create or Open Logging files
    train_csv = LogFile(os.path.join(exp_dir, 'train.csv'), args)
    test_csv = LogFile(os.path.join(exp_dir, 'test.csv'), args)
    best_txt = os.path.join(exp_dir, 'best.txt')

    save_args(exp_dir, args)  # Save args to JSON file

############ TRAINING LOOP ############
    for epoch in range(start_epoch, args.epochs):
            print('\n==> Training Epoch [{}] (lr={})'.format(epoch, optimizer.param_groups[0]['lr']))

            train_err_avg = train_epoch(train_loader, model, optimizer, loss, epoch)

            # Learning rate scheduler
            lr_decayer.step()

            train_csv.update_log(train_err_avg, epoch)

            # Save checkpoint in case evaluation crashed
            save_checkpoint({
                'args': args,
                'epoch': epoch,
                'model': model,
                'best_result': best_result,
                'optimizer': optimizer,
            }, False, epoch, exp_dir)

            # Evaluate the trained epoch
            test_err_avg, out_image = evaluate_epoch(val_loader, model, loss, epoch)  # evaluate on validation set

            # Evaluate Uncerainty
            ause = None
            if args.eval_uncert:
                if args.loss == 'masked_prob_loss_var':
                    ause, ause_fig = eval_ause(model, val_loader, args, epoch, uncert_type='v')
                else:
                    ause, ause_fig = eval_ause(model, val_loader, args, epoch, uncert_type='c')

            # Log to tensorboard if enabled
            if tb_writer is not None:
                avg_meter = test_err_avg.get_avg()
                tb_writer.add_scalar('Loss/val', avg_meter.loss, epoch)
                tb_writer.add_scalar('MAE/val', avg_meter.metrics['mae'], epoch)
                tb_writer.add_scalar('RMSE/val', avg_meter.metrics['rmse'], epoch)
                if ause is not None:
                    tb_writer.add_scalar('AUSE/val', ause, epoch)
                tb_writer.add_images('Prediction', colored_depthmap_tensor(out_image[:, :1, :, :]), epoch)
                tb_writer.add_images('Input_Conf_Log_Scale', colored_depthmap_tensor(torch.log(out_image[:, 2:, :, :]+1)), epoch)
                tb_writer.add_images('Output_Conf_Log_Scale', colored_depthmap_tensor(torch.log(out_image[:, 1:2, :, :]+1)), epoch)
                tb_writer.add_figure('Sparsification_Plot', ause_fig, epoch)

            # Update Log files
            test_csv.update_log(test_err_avg, epoch, ause)

            # Save best model
            # TODO: How to decide the best based on dataset?
            is_best = test_err_avg.metrics['rmse'] < best_result.metrics['rmse']
            if is_best:
                best_result = test_err_avg  # Save the new best locally
                test_err_avg.print_to_txt(best_txt, epoch)  # Print to a text file

            # Save it again if it is best checkpoint
            save_checkpoint({
                    'args': args,
                    'epoch': epoch,
                    'model': model,
                    'best_result': best_result,
                    'optimizer': optimizer,
                }, is_best, epoch, exp_dir)
            # TODO: Do you really need to save the best out_image ??


############ TRAINING FUNCTION ############
def train_epoch(dataloader, model, optimizer, objective, epoch):
    """
    Training function 
    
    Args:
        dataloader: The dataloader object for the dataset
        model: The model to be trained
        optimizer: The optimizer to be used
        objective: The objective function
        epoch: What epoch to start from
    
    Returns:
        AverageMeter() object.
    
    Raises:
        KeyError: Raises an exception.
    """
    err = create_error_metric(args)
    err_avg = AverageMeter(err.get_metrics())  # Accumulator for the error metrics

    model.train()  # switch to train mode

    start = time.time()
    for i, (input, target) in enumerate(dataloader):
            input, target = input.to(device), target.to(device)

            torch.cuda.synchronize()  # Wait for all kernels to finish

            data_time = time.time() - start

            start = time.time()

            optimizer.zero_grad()  # Clear the gradients

            # Forward pass
            out = model(input)

            loss = objective(out, target)  # Compute the loss

            # Backward pass
            loss.backward()

            optimizer.step()  # Update the parameters

            gpu_time = time.time() - start

            # Calculate Error metrics
            err = create_error_metric(args)
            err.evaluate(out[:, :1, :, :].data, target.data)
            err_avg.update(err.get_results(), loss.item(), gpu_time, data_time, input.size(0))

            if (i + 1) % args.print_freq == 0 or i == len(dataloader)-1:
                print('[Train] Epoch ({}) [{}/{}]: '.format(
                    epoch, i+1, len(dataloader)),  end='')
                print(err_avg)

            # Log to Tensorboard if enabled
            if tb_writer is not None:
                if (i + 1) % tb_freq == 0:
                    avg_meter = err_avg.get_avg()
                    tb_writer.add_scalar('Loss/train', avg_meter.loss, epoch * len(dataloader) + i)
                    tb_writer.add_scalar('MAE/train', avg_meter.metrics['mae'], epoch * len(dataloader) + i)
                    tb_writer.add_scalar('RMSE/train', avg_meter.metrics['rmse'], epoch * len(dataloader) + i)

            start = time.time()  # Start counting again for the next iteration

    return err_avg


############ EVALUATION FUNCTION ############
def evaluate_epoch(dataloader, model, objective, epoch):
    """
    Evluation function
    
    Args:
        dataloader: The dataloader object for the dataset
        model: The model to be trained
        epoch: What epoch to start from
    
    Returns:
        AverageMeter() object.
    
    Raises:
        KeyError: Raises an exception.
    """
    print('\n==> Evaluating Epoch [{}]'.format(epoch))

    err = create_error_metric(args)
    err_avg = AverageMeter(err.get_metrics())  # Accumulator for the error metrics

    model.eval()  # Swith to evaluate mode

    # Save output images
    out_img_saver = create_out_image_saver(exp_dir, args, epoch)
    out_image = None

    start = time.time()
    with torch.no_grad(): # Disable gradients computations
        for i, (input, target) in enumerate(dataloader):
            input, target = input.to(device), target.to(device)

            torch.cuda.synchronize()

            data_time = time.time() - start

            # Forward Pass
            start = time.time()

            out = model(input)

            # Check if there is cout There is Cout
            loss = objective(out, target)  # Compute the loss

            gpu_time = time.time() - start

            # Calculate Error metrics
            err = create_error_metric(args)
            err.evaluate(out[:, :1, :, :].data, target.data)
            err_avg.update(err.get_results(), loss.item(), gpu_time, data_time, input.size(0))

            # Save output images
            if args.save_val_imgs:
                out_image = out_img_saver.update(i, out_image, input, out, target)

            if args.evaluate is None:
                if tb_writer is not None and i == 1:  # Retrun batch 1 for tensorboard logging
                    out_image = out

            if (i + 1) % args.print_freq == 0 or i == len(dataloader)-1:
                print('[Eval] Epoch: ({0}) [{1}/{2}]: '.format(
                    epoch, i + 1, len(dataloader)), end='')
                print(err_avg)

            start = time.time()

    return err_avg, out_image


if __name__ == '__main__':
    main()

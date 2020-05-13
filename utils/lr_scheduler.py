
from torch.optim import lr_scheduler


def create_lr_scheduler(opt, args, start_epoch):

    if args.lr_scheduler == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 - args.niter) / float(args.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(opt, lr_lambda=lambda_rule)

    elif args.lr_scheduler == 'step':
        scheduler = lr_scheduler.StepLR(opt, step_size=args.lr_decay_step, gamma=args.lr_decay_factor,
                                        last_epoch=start_epoch)

    elif args.lr_scheduler == 'plateau':
        return NotImplementedError('Pleateau Scheduler is not implemented!')
        #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
        #                                           factor=args.gamma,
         #                                          threshold=0.0001,
         #                                          patience=args.lr_decay_iters)
    elif args.lr_scheduler == 'none':
        scheduler = None

    return scheduler

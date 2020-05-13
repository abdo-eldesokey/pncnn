import os
import torch
import math
import csv
import numpy as np


kitti_metrics = ['mae', 'absrel', 'mse', 'rmse', 'imae',  'irmse',
                'delta1', 'delta2', 'delta3']
nyu_metrics = ['mae', 'absrel', 'mse', 'rmse', 'delta1', 'delta2', 'delta3']
flat_metrics = ['mae', 'mse', 'rmse']


def create_error_metric(args):
    if args.dataset == 'kitti_depth':
        return KittiErrorMetrics(args.norm_factor)
    elif args.dataset == 'nyudepthv2':
        return NyuErrorMetrics()
    else:
        raise ValueError('{} has no defined error metric!'.format(args.dataset))


############ ERROR METRICS ############
class ErrorMetrics(object):
    def __init__(self, metrics_list):
        self.metrics = dict.fromkeys(metrics_list)
        for key, value in self.metrics.items():
            self.metrics[key] = 0

    def get_results(self):
        return self.metrics

    def get_metrics(self):
        return list(self.metrics.keys())


class KittiErrorMetrics(ErrorMetrics):
    def __init__(self, _norm_factor):
        metrics_list = kitti_metrics
        self.norm_factor = _norm_factor

        super(KittiErrorMetrics, self).__init__(metrics_list)
        self.delta_thresh = 1.25

    def set_to_worst(self):
        self.metrics['mae'] = np.inf
        self.metrics['absrel'] = np.inf
        self.metrics['mse'] = np.inf
        self.metrics['rmse'] = np.inf
        self.metrics['imae'] = np.inf
        self.metrics['delta1'] = 0.
        self.metrics['delta2'] = 0.
        self.metrics['delta3'] = 0.

    def evaluate(self, pred, target):
        pred *= self.norm_factor / 256  # In meters now
        target *= self.norm_factor / 256

        # Mask out values that has no groundtruth
        valid_mask = target > 0
        pred = pred[valid_mask]
        target = target[valid_mask]

        pred_mm = pred * 1e3
        target_mm = target * 1e3

        abs_diff = (pred_mm - target_mm).abs()
        self.metrics['mae'] = float(abs_diff.mean())
        self.metrics['absrel'] = float((abs_diff / target_mm).mean())
        self.metrics['mse'] = float((torch.pow(abs_diff, 2)).mean())
        self.metrics['rmse'] = math.sqrt(self.metrics['mse'])

        max_ratio = torch.max(pred_mm / target_mm, target_mm / pred_mm)
        self.metrics['delta1'] = float((max_ratio < self.delta_thresh).float().mean()*100)
        self.metrics['delta2'] = float((max_ratio < self.delta_thresh**2).float().mean()*100)
        self.metrics['delta3'] = float((max_ratio < self.delta_thresh**3).float().mean()*100)

        inv_output_km = (1e-3 * pred)**(-1)
        inv_target_km = (1e-3 * target)**(-1)
        abs_inv_diff = (inv_output_km - inv_target_km).abs()
        self.metrics['irmse'] = math.sqrt((torch.pow(abs_inv_diff, 2)).mean())
        self.metrics['imae'] = float(abs_inv_diff.mean())


class NyuErrorMetrics(ErrorMetrics):
    def __init__(self):
        metrics_list = nyu_metrics
        super(NyuErrorMetrics, self).__init__(metrics_list)
        self.delta_thresh = 1.25

    def set_to_worst(self):
        self.metrics['mae'] = np.inf
        self.metrics['absrel'] = np.inf
        self.metrics['mse'] = np.inf
        self.metrics['rmse'] = np.inf
        self.metrics['delta1'] = 0.
        self.metrics['delta2'] = 0.
        self.metrics['delta3'] = 0.

    def items(self):
        return self.metrics

    def evaluate(self, pred, target):
        # Mask out values that has no groundtruth
        valid_mask = target > 0
        pred = pred[valid_mask]
        target = target[valid_mask]

        abs_diff = (pred - target).abs()
        self.metrics['mae'] = float(abs_diff.mean())
        self.metrics['absrel'] = float((abs_diff / target).mean())
        self.metrics['mse'] = float((torch.pow(abs_diff, 2)).mean())
        self.metrics['rmse'] = math.sqrt(self.metrics['mse'])

        max_ratio = torch.max(pred / target, target / pred)
        self.metrics['delta1'] = float((max_ratio < self.delta_thresh).float().mean()*100)
        self.metrics['delta2'] = float((max_ratio < self.delta_thresh**2).float().mean()*100)
        self.metrics['delta3'] = float((max_ratio < self.delta_thresh**3).float().mean()*100)


############ AVERGAE METER ############
class AverageMeter(object):
    def __init__(self, error_metrics):
        self.metrics = dict.fromkeys(error_metrics)
        self.cnt = 0
        self.gpu_time = 0
        self.data_time = 0
        self.loss = 0
        self.init()

    def init(self):
        for key, value in self.metrics.items():
            self.metrics[key] = 0

    def update(self, new_values, loss, gpu_time, data_time, n=1):
        self.cnt += n

        for key, value in self.metrics.items():
            self.metrics[key] += (new_values[key] * n)

        self.loss += loss * n
        self.gpu_time += gpu_time * n
        self.data_time += data_time * n

    def get_avg(self):
        import copy
        avg = copy.deepcopy(self)

        avg.loss /= avg.cnt
        for key, value in avg.metrics.items():
            avg.metrics[key] /= avg.cnt
        avg.gpu_time /= avg.cnt
        avg.data_time /= avg.cnt

        return avg

    def __str__(self):
        # Print the average
        avg = self.get_avg()
        st = str()
        st += 'Loss: {:.3f},  '.format(avg.loss)
        for key, value in avg.metrics.items():
            st += str.swapcase(key)
            st += ': '
            st += '{:.3f},  '.format(value)

        st += 'GPU_time: '
        st += '{:.3f}  '.format(avg.gpu_time)
        return st

    # Used to save best.txt file
    def print_to_txt(self, txt_file, epoch):
        avg = self.get_avg()
        with open(txt_file, 'w') as txtfile:
            txtfile.write('Epoch: ({})\n===========\n'.format(epoch))
            txtfile.write('Loss: {:.5f}\n'.format(avg.loss))
            for key, value in avg.metrics.items():
                txtfile.write(str.swapcase(key) + ': {:.5f}\n'.format(value))

            txtfile.write('GPU_time: {:.5f}\n'.format(avg.gpu_time))
            txtfile.write('DATA_time: {:.5f}\n'.format(avg.data_time))


############ LOGFILE ############
class LogFile(object):
    def __init__(self, path_to_file, args):
        self.path_to_file = path_to_file

        # Define error metrics for the dataset
        if args.dataset == 'kitti_depth' or args.dataset == 'vkitti' or args.dataset == 'kitti_odo':
            self.field_names = kitti_metrics.copy()
        elif args.dataset == 'nyudepthv2' :
            self.field_names = nyu_metrics.copy()
        else:
            raise(ValueError('No field names defined for: {}'.format(args.dataset)))

        self.field_names.insert(0, 'epoch')  # Add an entry for epoch number
        self.field_names.insert(1, 'loss')  # Add an entry for epoch number
        self.field_names.append('gpu_time')
        self.field_names.append('data_time')
        if args.eval_uncert:
            self.field_names.append('ause')

        # Create the csv files
        if not os.path.isfile(path_to_file):
            with open(path_to_file, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.field_names)
                writer.writeheader()

    def update_log(self, avg_meter, epoch, ause=None):
        avg = avg_meter.get_avg()
        with open(self.path_to_file, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.field_names)

            if ause is None:
                writer.writerow({**{'epoch': epoch}, **{'loss': avg.loss}, **avg.metrics,
                                **{'gpu_time': avg.gpu_time, 'data_time': avg.data_time}})
            else:
                writer.writerow({**{'epoch': epoch}, **{'loss': avg.loss}, **avg.metrics,
                                 **{'gpu_time': avg.gpu_time, 'data_time': avg.data_time, 'ause': ause}})


############ MAIN FUNCTION ############
if __name__ == '__main__':

    # Test ErrorMetrics and Average Meter
    err = create_error_metric('nyudepthv2')
    avg = AverageMeter(err.get_metrics())

    import time as t
    start = t.time()
    err.evaluate(torch.rand((4, 1, 1200, 300)).cuda(), torch.rand((4, 1, 1200, 300)).cuda())
    print(t.time() - start)

    avg.update(err.get_results(), 1, 5, 4)

    print(avg.__str__())

    err = create_error_metric('kitti_depth')
    err.evaluate(torch.rand((4, 1, 1200, 300)).cuda(),torch.rand((4, 1, 1200, 300)).cuda())

    avg.update(err.get_results(), 1, 5, 4)

    print(avg.__str__())

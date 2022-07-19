import os.path as osp
import pickle
import shutil
import tempfile

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info
import torch.nn.functional as F
from mmcv.utils import print_log
import numpy as np
import os
from mmselfsup.datasets.semantic_kitti import augmentation_random_flip, random_rotate_pc, get_mask, mask_op


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class IouEval:
    def __init__(self, n_classes, ignore=None):
        # classes
        self.n_classes = n_classes

        # What to include and ignore from the means
        self.ignore = np.array(ignore, dtype=np.int64)
        self.include = np.array(
            [n for n in range(self.n_classes) if n not in self.ignore], dtype=np.int64)
        print("[IOU EVAL] IGNORE: ", self.ignore)
        print("[IOU EVAL] INCLUDE: ", self.include)

        # reset the class counters
        self.reset()

    def num_classes(self):
        return self.n_classes

    def reset(self):
        self.conf_matrix = np.zeros((self.n_classes,
                                     self.n_classes),
                                    dtype=np.int64)

    def addBatch(self, x, y):  # x=preds, y=targets
        # sizes should be matching
        x_row = x.reshape(-1)  # de-batchify
        y_row = y.reshape(-1)  # de-batchify

        # check
        assert (x_row.shape == x_row.shape)

        # create indexes
        idxs = tuple(np.stack((x_row, y_row), axis=0))

        # make confusion matrix (cols = gt, rows = pred)
        np.add.at(self.conf_matrix, idxs, 1)

    def getStats(self):
        # remove fp from confusion on the ignore classes cols
        conf = self.conf_matrix.copy()
        conf[:, self.ignore] = 0

        # get the clean stats
        tp = np.diag(conf)
        fp = conf.sum(axis=1) - tp
        fn = conf.sum(axis=0) - tp
        return tp, fp, fn

    def getIoU(self):
        tp, fp, fn = self.getStats()
        intersection = tp
        union = tp + fp + fn + 1e-15
        iou = intersection / union
        iou_mean = (intersection[self.include] / union[self.include]).mean()
        return iou_mean, iou  # returns "iou mean", "iou per class" ALL CLASSES

    def getacc(self):
        tp, fp, fn = self.getStats()
        total_tp = tp.sum()
        total = tp[self.include].sum() + fp[self.include].sum() + 1e-15
        acc_mean = total_tp / total
        return acc_mean  # returns "acc mean"

    def get_confusion(self):
        return self.conf_matrix.copy()


def single_gpu_test(model, data_loader, show=False):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, **data)
        results.append(result)

        if show:
            model.module.show_result(data, result)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def single_semantic_kitti_test(model, data_loader, logger,
                               gpu_collect=True):
    model.eval()
    results = []
    dataset = data_loader.dataset
    evaluator = IouEval(dataset.get_n_classes(),
                        dataset.ignore_class)
    acc = AverageMeter()
    iou = AverageMeter()
    progress = 0
    rank, world_size = get_dist_info()
    print("world size is ", world_size)
    evaluation_status = []
    conf_matrix = np.zeros((dataset.get_n_classes(),
                            dataset.get_n_classes()),
                           dtype=np.int64)
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)
            output = F.softmax(result, dim=1)
            argmax = output.argmax(dim=1)
            # argmax back to 0-19
            argmax_algo_1 = (argmax + 1).clone()
            # print(data['label'], data['scan'][0].size(0))
            evaluator.addBatch(argmax_algo_1.data.cpu().numpy(), data['label'].data[0])
            if rank == 0:
                if 100 * i / (len(dataset)) > progress:
                    print("{:d}% ".format(progress), end="", flush=True)
                    progress += 10
            # batch_size = data['scan'][0].size(0)
    # if gpu_collect:
    #     evaluation_status = conf_matrix  # collect_results_gpu([torch.from_numpy(conf_matrix).long().cuda()], world_size)
    # if rank == 0:
    accuracy = evaluator.getacc()
    jaccard, class_jaccard = evaluator.getIoU()
    acc.update(accuracy.item(), 1)
    iou.update(jaccard.item(), 1)
    results = {}
    for i, jacc in enumerate(class_jaccard):
        # print('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
        #     i=i, class_str=class_func(i), jacc=jacc))
        print_log('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
            i=i, class_str=dataset.get_xentropy_class_string(i), jacc=jacc), logger=logger)
        results[dataset.get_xentropy_class_string(i)] = jacc

    print_log('Validation set:\n'
              'Acc avg {acc.avg:.3f}\n'
              'IoU avg {iou.avg:.3f}'.format(
        acc=acc, iou=iou))
    results["acc"] = acc.avg
    results["iou"] = iou.avg
    return results
    # else:
    #     return None

def multigpu_semantic_kitti_test(model, data_loader, logger,
                                 gpu_collect=True, save_scan=False):
    model.eval()
    results = []
    dataset = data_loader.dataset
    evaluator = IouEval(dataset.get_n_classes(),
                        dataset.ignore_class)
    acc = AverageMeter()
    iou = AverageMeter()
    progress = 0
    rank, world_size = get_dist_info()
    print("world size is ", world_size)
    evaluation_status = []
    conf_matrix = np.zeros((dataset.get_n_classes(),
                            dataset.get_n_classes()),
                            dtype=np.int64)
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)
            output = F.softmax(result["x"], dim=1)
            argmax = output.argmax(dim=1)
            # argmax back to 0-19
            argmax_algo_1 = (argmax + 1).clone()
            # print(data['label'], data['scan'][0].size(0))
            # evaluator.addBatch(argmax_algo_1.data.cpu().numpy(), data['label'].data[0])
            x_row = argmax_algo_1.data.cpu().numpy().reshape(-1)  # de-batchify
            y_row = data['points_label'].data[0][0].reshape(-1)  # de-batchify

            # check
            assert (x_row.shape == x_row.shape)

            # create indexes
            idxs = tuple(np.stack((x_row, y_row), axis=0))

            # make confusion matrix (cols = gt, rows = pred)
            np.add.at(conf_matrix, idxs, 1)
            if rank == 0:
                if 100 * i / (len(dataset) / world_size) > progress:
                    print("{:d}% ".format(progress), end="", flush=True)
                    progress += 10
            # batch_size = data['scan'][0].size(0)
    if gpu_collect:
        evaluation_status = collect_results_cpu([torch.from_numpy(conf_matrix).long()], world_size)
    if rank == 0:
        merge_result = evaluation_status[0].data.numpy()
        for i in range(1, len(evaluation_status)):
            merge_result += evaluation_status[i].data.numpy()
        evaluator.conf_matrix = merge_result
        accuracy = evaluator.getacc()
        jaccard, class_jaccard = evaluator.getIoU()
        acc.update(accuracy.item(), 1)
        iou.update(jaccard.item(), 1)
        results = {}
        for i, jacc in enumerate(class_jaccard):
            # print('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
            #     i=i, class_str=class_func(i), jacc=jacc))
            print_log('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
                i=i, class_str=dataset.get_xentropy_class_string(i), jacc=jacc), logger=logger)
            results[dataset.get_xentropy_class_string(i)] = jacc

        print_log('Validation set:\n'
                'Acc avg {acc.avg:.3f}\n'
                'IoU avg {iou.avg:.3f}'.format(
            acc=acc, iou=iou))
        results["acc"] = acc.avg
        results["iou"] = iou.avg
        return results
    else:
        return None


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.append(result)

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results

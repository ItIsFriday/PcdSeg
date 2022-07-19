import os.path as osp

from mmcv.runner import Hook
from torch.utils.data import DataLoader

class KittiSegEvalHook(Hook):
    """Evaluation hook.

    Attributes:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by epochs). Default: 1.
    """

    def __init__(self, dataloader, interval=1, **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError(
                'dataloader must be a pytorch DataLoader, but got {}'.format(
                    type(dataloader)))
        self.dataloader = dataloader
        self.interval = interval
        self.eval_kwargs = eval_kwargs

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        from mmselfsup.apis import single_semantic_kitti_test, multigpu_semantic_kitti_test
        multi_gpu_test_for_kitti = True
        if multi_gpu_test_for_kitti:
            results = multigpu_semantic_kitti_test(runner.model, self.dataloader, runner.logger)
            if runner.rank == 0:
                for name, val in results.items():
                    runner.log_buffer.output[name] = val
                runner.log_buffer.ready = True
        else:
            results = single_semantic_kitti_test(runner.model, self.dataloader, runner.logger)
            # if runner.rank == 0:
            for name, val in results.items():
                runner.log_buffer.output[name] = val
            runner.log_buffer.ready = True

    def evaluate(self, runner, results):
        eval_res = self.dataloader.dataset.evaluate(
            results, logger=runner.logger, **self.eval_kwargs)
        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True

# Copyright (c) OpenMMLab. All rights reserved.
from .train import init_random_seed, set_random_seed, train_model
from .test import multi_gpu_test, single_gpu_test, single_semantic_kitti_test, \
    multigpu_semantic_kitti_test

__all__ = ['init_random_seed', 'set_random_seed', 'train_model',
'single_semantic_kitti_test', 'multigpu_semantic_kitti_test'
]

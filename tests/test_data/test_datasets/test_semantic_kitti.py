# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmselfsup.datasets import SemanticKitti


def test_getitem():
    np.random.seed(0)
    root_path = './tests/data/semantickitti/'
    ann_file = './tests/data/semantickitti/semantickitti_infos.pkl'
    class_names = ('unlabeled', 'car', 'bicycle', 'motorcycle', 'truck', 'bus',
                   'person', 'bicyclist', 'motorcyclist', 'road', 'parking',
                   'sidewalk', 'other-ground', 'building', 'fence',
                   'vegetation', 'trunck', 'terrian', 'pole', 'traffic-sign')
    pipelines = [
        dict(type='DefaultFormatBundle3D', class_names=class_names),
        dict(
            type='Collect3D',
            keys=[
                'points'
            ],
            # meta_keys=['file_name', 'sample_idx', 'pcd_rotation']
            ),
    ]

    semantickitti_dataset = SemanticKitti("/home/deeproute/kitti_odometry/dataset/sequences",
                         "/home/deeproute/code/lidar-seg/train/tasks/semantic/config/labels/semantic-kitti.yaml",
                         'train', [[-100, 100], [-100, 100], [-3, 1.8]], pipelines, augmentation=False, with_gt=False,
                         shuffle_index=False, ignore_class=[0]

                                                 )
    data = semantickitti_dataset[0]
    print(data['points']._data.shape)

test_getitem()
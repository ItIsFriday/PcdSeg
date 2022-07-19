class_names = ('unlabeled', 'car', 'bicycle', 'motorcycle', 'truck', 'bus',
                   'person', 'bicyclist', 'motorcyclist', 'road', 'parking',
                   'sidewalk', 'other-ground', 'building', 'fence',
                   'vegetation', 'trunck', 'terrian', 'pole', 'traffic-sign')
pipelines = [
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=[
            'points_q', 'points_k'
        ],
        # meta_keys=['file_name', 'sample_idx', 'pcd_rotation']
        ),
]
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # useless in pointcloud

prefetch = False
data = dict(
    samples_per_gpu=2,  # total 32*8(gpu)=256
    workers_per_gpu=4,
    train=dict(
        type="SemanticKitti",
        data_root="/home/deeproute/kitti_odometry/dataset/sequences",
        data_config_file="/home/deeproute/code/lidar-seg/train/tasks/semantic/config/labels/semantic-kitti.yaml",
        setname='train', 
        lims=[[-48, 48], [-48, 48], [-3, 1.8]],
        pipelines=pipelines,
        prefetch = prefetch,
        pre_training=True,
        augmented_dir="/home/deeproute/kitti_odometry/segcontrast/"
    ))

# find_unused_parameters=True
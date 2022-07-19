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
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # useless in pointcloud
lims = [[-48, 48], [-48, 48], [-3, 1.8]]
prefetch = False
data = dict(
    samples_per_gpu=2,  # total 32*8(gpu)=256
    workers_per_gpu=4,
    train=dict(
        type="SemanticKitti",
        data_root="/home/deeproute/kitti_odometry/dataset/sequences",
        data_config_file="/home/deeproute/code/lidar-seg/train/tasks/semantic/config/labels/semantic-kitti.yaml",
        setname='train', 
        lims=lims,
        pipelines=pipelines,
        prefetch = prefetch
    ))
voxel_size = [480, 480, 48]
# model settings
model = dict(
    type='SparseUNet',
    voxel_layer=dict(
        lims=lims,
        voxel_size=voxel_size),
    backbone=dict(
        type='MinkUNet',
        num_classes=19,
        in_channels=4,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='BN')))


# checkpoint saving
checkpoint_config = dict(interval=10)

# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])
# yapf:enable

# runtime settings
dist_params = dict(backend='nccl')
cudnn_benchmark = True
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
persistent_workers = True

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
# optimizer
optimizer = dict(type='AdamW', lr=1e-3, betas=(0.9, 0.999), weight_decay=0.05)
optimizer_config = dict(grad_clip=dict(max_norm=5.0))

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0.,
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=1e-4,  # cannot be 0
    warmup_by_epoch=True)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=100)
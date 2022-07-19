class_names = ('unlabeled', 'car', 'bicycle', 'motorcycle', 'truck', 'bus',
                   'person', 'bicyclist', 'motorcyclist', 'road', 'parking',
                   'sidewalk', 'other-ground', 'building', 'fence',
                   'vegetation', 'trunck', 'terrian', 'pole', 'traffic-sign')
pipelines = [
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=[
            'points', 'points_label', 'filter_mask'
        ],
        # meta_keys=['file_name', 'sample_idx', 'pcd_rotation']
        ),
]
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # useless in pointcloud
lims = [[-48, 48], [-48, 48], [-3, 1.8]]
prefetch = False
data = dict(
    samples_per_gpu=3,  # total 32*8(gpu)=256
    workers_per_gpu=4,
    train=dict(
        type="SemanticKitti",
        data_root="/home/deeproute/kitti_odometry/dataset/sequences",
        data_config_file="./configs/kitti.yaml",
        setname='train', 
        lims=lims,
        pipelines=pipelines,
        prefetch = prefetch,
        with_gt=True,
        augmentation=True
    ),

    val=dict(
        type="SemanticKitti",
        data_root="/home/deeproute/kitti_odometry/dataset/sequences",
        data_config_file="./configs/kitti.yaml",
        setname='valid', 
        lims=lims,
        pipelines=pipelines,
        prefetch = prefetch,
        with_gt=True
    ),
    
    )
voxel_size = [1920, 1920, 96]
# model settings
model = dict(
    type='GASN',
    train_cfg = dict(
    lims=[[-48, 48], [-48, 48], [-3, 1.8]],
    offset=0.5,
    target_scale=1,
    grid_meters=[0.2, 0.2, 0.1],
    scales=[0.5, 1],
    pooling_scale=[0.5, 1, 2, 4, 6, 8, 12],
    sizes=[480, 480, 48],
    n_class=19
))


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
# load_from = "work_dirs/moco_0_0_5/backbone.pth"
# load_from = "work_dirs/test_mae/output.pth"
load_from = None
resume_from = None
workflow = [('train', 1)]
persistent_workers = True

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
# optimizer
# optimizer = dict(type='SGD', lr=2.4e-1, weight_decay=1e-4, momentum=0.9, nesterov=True)
optimizer = dict(type='Adam', lr=0.002, weight_decay=1e-4)
# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
optimizer_config = dict()
# learning policy
find_unused_parameters=True
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=50,
    warmup_ratio=1.0 / 3,
    gamma=0.25,
    step=[15, 25, 35, 45])
# lr_config = dict(policy='CosineAnnealing', min_lr=0.24/1000)
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=50)
evaluation = dict(do=True)
# load_from="work_dirs/segcontrast/latest.pth"
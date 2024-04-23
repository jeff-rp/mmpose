_base_ = ['mmpose::_base_/default_runtime.py']

# runtime
max_epochs = 20
base_lr = 8e-4
train_batch_size = 96
accumulative_counts = 1
val_batch_size = 64
num_workers = 6
val_interval = 1
cos_annealing_begin = 5
data_root = '../'
backbone_checkpoint = 'https://download.openmmlab.com/mmclassification/v0/mobileone/'\
                      'mobileone-s1_8xb32_in1k_20221110-ceeef467.pth'
head_checkpoint = None

train_cfg = dict(max_epochs=max_epochs, val_interval=val_interval)
randomness = dict(seed=21)

# optimizer
optim_wrapper = dict(type='OptimWrapper',
                     optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
                     paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True),
                     accumulative_counts=accumulative_counts)

# learning policy
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-5, by_epoch=False,
         begin=0, end=int(1000*(256/train_batch_size))),
    dict(type='CosineAnnealingLR', eta_min=base_lr * 0.05, begin=cos_annealing_begin, end=max_epochs,
         T_max=max_epochs-cos_annealing_begin, by_epoch=True, convert_to_iter_based=True)
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=1024)

# codec settings
codec = dict(type='UDPHeatmap', input_size=(192, 192), heatmap_size=(48, 48), sigma=1.5)

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[127.5, 127.5, 127.5],
        std=[255.0, 255.0, 255.0],
        bgr_to_rgb=True
    ),
    backbone=dict(
        _scope_='mmpretrain',
        type='MobileOne',
        arch='s1',
        out_indices=(0, 1, 2, 3),
        init_cfg=dict(type='Pretrained', prefix='backbone.', checkpoint=backbone_checkpoint)
    ),
    head=dict(
        type='FFHead',
        in_channels=(96, 192, 512, 1280),
        out_channels=21,
        arch_type='C',
        up_linear=True,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec,
        init_cfg=dict(type='Pretrained', prefix='head.', checkpoint=head_checkpoint)
                 if head_checkpoint is not None else None
    ),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=False
    )
)

# pipelines
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomBBoxTransform', scale_factor=[0.7, 1.4], rotate_factor=180),
    # dict(type='RandomBBoxTransform', shift_prob=0.0, scale_prob=0.0, rotate_prob=0.0),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='TopdownAffine', input_size=codec['input_size'], use_udp=True),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='Albumentation', transforms=[
        dict(type='Blur', p=0.1),
        dict(type='MedianBlur', p=0.1),
        dict(type='CoarseDropout', max_holes=1, max_height=0.4, max_width=0.4, min_holes=1,
             min_height=0.2, min_width=0.2, p=0.5)
    ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

# train datasets
dataset_freihand = dict(
    type='RepeatDataset',
    dataset=dict(
        type='FreiHandDataset',
        data_root=data_root+'freihand/',
        data_mode='topdown',
        ann_file='annotations/freihand_train.json',
        data_prefix=dict(img=''),
        # indices=256,
        pipeline=[]
    ),
    times=3
)

dataset_uhand = dict(
    type='RepeatDataset',
    dataset=dict(
        type='OneHand10KDataset',
        data_root=data_root+'uhand/',
        data_mode='topdown',
        ann_file='train.json',
        data_prefix=dict(img=''),
        # indices=256,
        pipeline=[]
    ),
    times=3
)

dataset_dart = dict(
    type='OneHand10KDataset',
    data_root=data_root+'DARTset/',
    data_mode='topdown',
    ann_file='annotations/train.json',
    data_prefix=dict(img=''),
    # indices=256,
    pipeline=[]
)

# dataset_freihand = dict(
#     type='FreiHandDataset',
#     data_root=data_root+'freihand/',
#     data_mode='topdown',
#     ann_file='annotations/freihand_train.json',
#     data_prefix=dict(img=''),
#     pipeline=[]
# )

# dataset_rhd = dict(
#     type='Rhd2DDataset',
#     data_root=data_root+'rhd/',
#     data_mode='topdown',
#     ann_file='annotations/rhd_train.json',
#     data_prefix=dict(img=''),
#     pipeline=[
#         dict(type='KeypointConverter', num_keypoints=21,
#              mapping=[(0, 0), (1, 4), (2, 3), (3, 2), (4, 1), (5, 8), (6, 7), (7, 6), (8, 5),
#                       (9, 12), (10, 11), (11, 10), (12, 9), (13, 16), (14, 15), (15, 14), (16, 13),
#                       (17, 20), (18, 19), (19, 18), (20, 17)])
#     ]
# )

# dataset_panoptic = dict(
#     type='PanopticHand2DDataset',
#     data_root=data_root+'panoptic/',
#     data_mode='topdown',
#     ann_file='annotations/panoptic_train.json',
#     data_prefix=dict(img=''),
#     pipeline=[
#         dict(type='KeypointConverter', num_keypoints=21,
#              mapping=[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8),
#                       (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16),
#                       (17, 17), (18, 18), (19, 19), (20, 20)])
#     ]
# )

# data loaders
train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CombinedDataset',
        metainfo=dict(from_file='configs/_base_/datasets/freihand2d.py'),
        datasets=[dataset_freihand, dataset_uhand, dataset_dart],
        pipeline=train_pipeline,
        test_mode=False
    )
)

# test datasets
val_freihand = dict(
    type='FreiHandDataset',
    data_root=data_root+'freihand/',
    data_mode='topdown',
    ann_file='annotations/freihand_val.json',
    data_prefix=dict(img=''),
    # indices=256,
    pipeline=[]
)

val_uhand = dict(
    type='OneHand10KDataset',
    data_root=data_root+'uhand/',
    data_mode='topdown',
    ann_file='test.json',
    data_prefix=dict(img=''),
    # indices=256,
    pipeline=[]
)

val_dart = dict(
    type='OneHand10KDataset',
    data_root=data_root+'DARTset/',
    data_mode='topdown',
    ann_file='annotations/test.json',
    data_prefix=dict(img=''),
    # indices=256,
    pipeline=[]
)

val_dataloader = dict(
    batch_size=val_batch_size,
    num_workers=num_workers//2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='CombinedDataset',
        metainfo=dict(from_file='configs/_base_/datasets/freihand2d.py'),
        datasets=[val_freihand, val_uhand, val_dart],
        pipeline=val_pipeline,
        test_mode=True
    )
    # dataset=dict(
    #     type='FreiHandDataset',
    #     data_root=data_root+'freihand/',
    #     data_mode='topdown',
    #     ann_file='annotations/freihand_val.json',
    #     data_prefix=dict(img=''),
    #     pipeline=val_pipeline,
    #     test_mode=True
    # )
)
test_dataloader = val_dataloader

# hooks
default_hooks = dict(
    checkpoint=dict(interval=1, save_best='AUC', rule='greater', max_keep_ckpts=1),
    logger=dict(interval=1000)
)
custom_hooks = [
    dict(type='EMAHook', ema_type='ExpMomentumEMA', momentum=0.0002, update_buffers=True, priority=49)
]

# evaluators
val_evaluator = [dict(type='PCKAccuracy', thr=0.2), dict(type='AUC'), dict(type='EPE')]
test_evaluator = val_evaluator
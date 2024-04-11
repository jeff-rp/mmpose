_base_ = ['mmpose::_base_/default_runtime.py']

# runtime
max_epochs = 210
base_lr = 0.0005
train_batch_size = 64
accumulative_counts = 1
val_batch_size = 32
num_workers = 4
val_interval = 10
cos_annealing_begin = 100
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
codec = dict(type='UDPHeatmap', input_size=(256, 256), heatmap_size=(64, 64), sigma=2)

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
    # head=dict(
    #     type='FFHead',
    #     in_channels=(96, 192, 512, 1280),
    #     out_channels=59,
    #     arch_type='C',
    #     up_linear=True,
    #     decoder=codec,
    #     init_cfg=dict(type='Pretrained', prefix='head.', checkpoint=head_checkpoint)
    #              if head_checkpoint is not None else None
    # ),
    head=dict(
        type='VisPredictHead',
        loss=dict(
            type='BCELoss',
            use_target_weight=True,
            use_sigmoid=True,
            loss_weight=1e-3,
        ),
        pose_cfg=dict(
            type='FFHead',
            in_channels=(96, 192, 512, 1280),
            out_channels=59,
            arch_type='C',
            loss=dict(type='KeypointMSELoss', use_target_weight=True, use_heatmap_weight=True),
            decoder=codec,
            init_cfg=dict(type='Pretrained', prefix='head.', checkpoint=head_checkpoint)
                          if head_checkpoint is not None else None
        )
    ),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=True
    )
)

# pipelines
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody',
         min_total_keypoints=10, min_upper_keypoints=3, upper_prioritized_prob=1.0),
    dict(type='RandomBBoxTransform', scale_factor=[0.7, 1.3], rotate_factor=70),
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
train_datasets = [
    dict(
        type='Coco59Dataset',
        data_root=data_root+'coco/',
        data_mode='topdown',
        ann_file='annotations/coco_body59_train.json',
        data_prefix=dict(img='train2017/'),
        pipeline=[]
    ),
    dict(
        type='Coco59Dataset',
        data_root=data_root+'UBody/',
        data_mode='topdown',
        ann_file='annotations/ubody59_train.json',
        pipeline=[],
        sample_interval=2
    )
]

# data loaders
train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CombinedDataset',
        metainfo=dict(from_file='configs/_base_/datasets/coco59.py'),
        datasets=train_datasets,
        pipeline=train_pipeline,
        test_mode=False
    )
)
val_dataloader = dict(
    batch_size=val_batch_size,
    num_workers=num_workers//2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='Coco59Dataset',
        data_root=data_root+'coco/',
        data_mode='topdown',
        ann_file='annotations/coco_body59_val.json',
        data_prefix=dict(img='val2017/'),
        pipeline=val_pipeline,
        test_mode=True
    )
)
test_dataloader = val_dataloader

# hooks
default_hooks = dict(
    checkpoint=dict(save_best='coco/AP', rule='greater', max_keep_ckpts=1),
    logger=dict(interval=100)
)
custom_hooks = [
    dict(type='EMAHook', ema_type='ExpMomentumEMA', momentum=0.0002, update_buffers=True, priority=49)
]

# evaluators
val_evaluator = dict(type='CocoMetric',
                     ann_file=data_root+'coco/annotations/coco_body59_val.json')
test_evaluator = val_evaluator
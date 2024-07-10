_base_ = ['mmpose::_base_/default_runtime.py']

# runtime
max_epochs = 100
base_lr = 0.001
train_batch_size = 64
val_batch_size = 32
num_workers = 0
val_interval = 10
data_root = '../'

train_cfg = dict(max_epochs=max_epochs, val_interval=val_interval)
randomness = dict(seed=21)

# optimizer
optim_wrapper = dict(type='OptimWrapper',
                     optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
                     paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# learning policy
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-5, by_epoch=False,
         begin=0, end=100),
]

# codec settings
codec = dict(type='RegressionLabel', input_size=(192, 192))

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[127.5, 127.5, 127.5],
        std=[255.0, 255.0, 255.0],
        bgr_to_rgb=True
    ),
    backbone=dict(type='MobileNetV2', widen_factor=1., out_indices=(7, )),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='RegressionHead',
        in_channels=1280,
        num_joints=21,
        loss=dict(type='SmoothL1Loss', use_target_weight=True),
        decoder=codec
    ),
    test_cfg=dict(
        flip_test=True,
        shift_coords=True
    )
)

# pipelines
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomBBoxTransform', scale_factor=[0.7, 1.4], rotate_factor=180),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='TopdownAffine', input_size=codec['input_size'], use_udp=True),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

# data loaders
train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=num_workers,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoHand3DDataset',
        data_root=data_root+'DARTset/',
        data_mode='topdown',
        ann_file='annotations/train.json',
        data_prefix=dict(img=''),
        pipeline=train_pipeline
    )
)
val_dataloader = dict(
    batch_size=val_batch_size,
    num_workers=num_workers,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='CocoHand3DDataset',
        data_root=data_root+'DARTset/',
        data_mode='topdown',
        ann_file='annotations/test.json',
        data_prefix=dict(img=''),
        pipeline=val_pipeline
    )
)
test_dataloader = val_dataloader

# hooks
default_hooks = dict(
    checkpoint=dict(interval=val_interval, save_best='AUC', rule='greater', max_keep_ckpts=1),
)

# evaluators
val_evaluator = [dict(type='PCKAccuracy', thr=0.2), dict(type='AUC'), dict(type='EPE')]
test_evaluator = val_evaluator
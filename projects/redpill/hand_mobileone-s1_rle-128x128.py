_base_ = ['mmpose::_base_/default_runtime.py']

# runtime
max_epochs = 150
base_lr = 0.001
train_batch_size = 256
accumulative_counts = 1
val_batch_size = 128
num_workers = 8
val_interval = 10
cos_annealing_begin = 50
data_root = '../'
backbone_checkpoint = 'work_dirs/hand_mobileone-s1_rle-160x160/best_AUC_epoch_140.pth'
head_checkpoint = None
log_interval=1000

train_cfg = dict(max_epochs=max_epochs, val_interval=val_interval)
randomness = dict(seed=21)

# optimizer
optim_wrapper = dict(type='OptimWrapper',
                     optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
                     paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True),
                     clip_grad=dict(max_norm=35, norm_type=2),
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
codec = dict(type='RegressionLabel', input_size=(128, 128))

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
        out_indices=(3,),
        init_cfg=dict(type='Pretrained', prefix='backbone.', checkpoint=backbone_checkpoint)
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='RLEHead',
        in_channels=1280,
        num_joints=21,
        loss=dict(type='RLELoss', use_target_weight=True),
        decoder=codec,
        init_cfg=dict(type='Pretrained', prefix='head.', checkpoint=head_checkpoint)
                 if head_checkpoint is not None else None
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
        type='CocoWholeBodyHandDataset',
        data_root=data_root+'coco/',
        data_mode='topdown',
        ann_file='annotations/cocow_hand_train_v2.json',
        data_prefix=dict(img='train2017/'),
        pipeline=[]
    ),
    dict(
        type='CocoWholeBodyHandDataset',
        data_root=data_root+'halpe/',
        data_mode='topdown',
        ann_file='annotations/halpe_hand_train_v2.json',
        data_prefix=dict(img='hico_20160224_det/images/train2015/'),
        pipeline=[],
    ),
    dict(
        type='OneHand10KDataset',
        data_root=data_root+'onehand10k/',
        data_mode='topdown',
        ann_file='annotations/train.json',
        data_prefix=dict(img=''),
        pipeline=[],
    ),
    dict(
        type='OneHand10KDataset',
        data_root=data_root+'freihand/',
        data_mode='topdown',
        ann_file='annotations/train.json',
        data_prefix=dict(img=''),
        pipeline=[],
    ),
    dict(
        type='OneHand10KDataset',
        data_root=data_root+'panoptic/',
        data_mode='topdown',
        ann_file='annotations/train.json',
        data_prefix=dict(img=''),
        pipeline=[],
    ),
    dict(
        type='OneHand10KDataset',
        data_root=data_root+'DARTset/',
        data_mode='topdown',
        ann_file='annotations/train.json',
        data_prefix=dict(img=''),
        pipeline=[],
        sample_interval=5
    ),
    dict(
        type='CocoWholeBodyHandDataset',
        data_root=data_root+'UBody/',
        data_mode='topdown',
        ann_file='annotations/hand_train_v2.json',
        data_prefix=dict(img=''),
        pipeline=[],
        sample_interval=3
    ),
    dict(
        type='OneHand10KDataset',
        data_root=data_root+'HInt/',
        data_mode='topdown',
        ann_file='annotations/train.json',
        data_prefix=dict(img=''),
        pipeline=[],
    )
]

# validation datasets
val_datasets = [
    dict(
        type='CocoWholeBodyHandDataset',
        data_root=data_root+'coco/',
        data_mode='topdown',
        ann_file='annotations/cocow_hand_val_v2.json',
        data_prefix=dict(img='val2017/'),
        pipeline=[]
    ),
    dict(
        type='OneHand10KDataset',
        data_root=data_root+'onehand10k/',
        data_mode='topdown',
        ann_file='annotations/val.json',
        data_prefix=dict(img=''),
        pipeline=[]
    ),
    dict(
        type='OneHand10KDataset',
        data_root=data_root+'freihand/',
        data_mode='topdown',
        ann_file='annotations/val.json',
        data_prefix=dict(img=''),
        pipeline=[]
    ),
    dict(
        type='OneHand10KDataset',
        data_root=data_root+'panoptic/',
        data_mode='topdown',
        ann_file='annotations/val.json',
        data_prefix=dict(img=''),
        pipeline=[]
    ),
    dict(
        type='OneHand10KDataset',
        data_root=data_root+'HInt/',
        data_mode='topdown',
        ann_file='annotations/val.json',
        data_prefix=dict(img=''),
        pipeline=[]
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
        metainfo=dict(from_file='configs/_base_/datasets/coco_wholebody_hand.py'),
        datasets=train_datasets,
        pipeline=train_pipeline,
        test_mode=False
    )
)
val_dataloader = dict(
    batch_size=val_batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='CombinedDataset',
        metainfo=dict(from_file='configs/_base_/datasets/coco_wholebody_hand.py'),
        datasets=val_datasets,
        pipeline=val_pipeline,
        test_mode=True
    )
)
test_dataloader = val_dataloader

# hooks
default_hooks = dict(
    checkpoint=dict(interval=1, save_best='AUC', rule='greater', max_keep_ckpts=1),
    logger=dict(interval=log_interval)
)
custom_hooks = [
    dict(type='EMAHook', ema_type='ExpMomentumEMA', momentum=0.0002, update_buffers=True, priority=49)
]

# evaluators
val_evaluator = [dict(type='PCKAccuracy', thr=0.2), dict(type='AUC'), dict(type='EPE')]
test_evaluator = val_evaluator
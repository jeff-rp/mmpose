_base_ = ['mmpose::_base_/default_runtime.py']

# runtime
max_epochs = 210
base_lr = 0.001
train_batch_size = 96
# accumulative_counts = 1
val_batch_size = 64
num_workers = 4
val_interval = 10
cos_annealing_begin = 70
data_root = '../'
backbone_checkpoint = None
head_checkpoint = None
log_interval=100

train_cfg = dict(max_epochs=max_epochs, val_interval=val_interval)
randomness = dict(seed=21)

# optimizer
optim_wrapper = dict(type='OptimWrapper',
                     optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
                     #clip_grad=dict(max_norm=35, norm_type=2),
                     paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))
                     #accumulative_counts=accumulative_counts)

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
custom_imports = dict(imports=['mmpretrain.models'], allow_failed_imports=False)
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[127.5, 127.5, 127.5],
        std=[255.0, 255.0, 255.0],
        bgr_to_rgb=True
    ),
    metainfo=dict(from_file='configs/_base_/datasets/coco_aic.py'),
    backbone=dict(
        #_delete_=True, # Delete the backbone field in _base_
        type='mmpretrain.TIMMBackbone', # Using timm from mmpretrain
        model_name='semnasnet_075.rmsp_in1k',
        features_only=True,
        pretrained=True if backbone_checkpoint is None else False,
        out_indices=(4,),
        init_cfg=dict(type='Pretrained', prefix='backbone.', checkpoint=backbone_checkpoint)
                 if backbone_checkpoint is not None else None
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=240,
        out_channels=19,
        loss=dict(type='KeypointMSELoss', use_target_weight=True, use_heatmap_weight=True),
        decoder=codec,
        init_cfg=dict(type='Pretrained', prefix='head.', checkpoint=head_checkpoint)
                 if head_checkpoint is not None else None
    ),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=True,
        output_keypoint_indices=list(range(17))
    )
)

# pipelines
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform', scale_factor=[0.7, 1.3], rotate_factor=70),
    dict(type='TopdownAffine', input_size=codec['input_size'], use_udp=True),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='Albumentation', transforms=[
        dict(type='Blur', p=0.1),
        dict(type='MotionBlur', p=0.1),
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
        type='CocoDataset',
        data_root=data_root+'coco/',
        data_mode='topdown',
        ann_file='annotations/person_keypoints_train2017.json',
        data_prefix=dict(img='train2017/'),
        pipeline=[dict(type='KeypointConverter', num_keypoints=19, mapping=[(i, i) for i in range(17)])]
    ),
    dict(
        type='AicDataset',
        data_root=data_root+'aic/',
        data_mode='topdown',
        ann_file='annotations/aic_train.json',
        data_prefix=dict(img='ai_challenger_keypoint_train_20170902/keypoint_train_images_20170902/'),
        pipeline=[dict(type='KeypointConverter',
                       num_keypoints=19,
                       mapping=[(0, 6), (1, 8), (2, 10), (3, 5), (4, 7), (5, 9), (6, 12), (7, 14),
                                (8, 16), (9, 11), (10, 13), (11, 15), (12, 17), (13, 18)])]
    )
]

# data loaders
train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(type='CombinedDataset',
                 metainfo=dict(from_file='configs/_base_/datasets/coco_aic.py'),
                 datasets=train_datasets,
                 pipeline=train_pipeline,
                 test_mode=False)
)
val_dataloader = dict(
    batch_size=val_batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(type='CocoDataset',
                 data_root=data_root+'coco/',
                 data_mode='topdown',
                 ann_file='annotations/person_keypoints_val2017.json',
                 data_prefix=dict(img='val2017/'),
                 pipeline=val_pipeline,
                 test_mode=True)
)
test_dataloader = val_dataloader

# hooks
default_hooks = dict(
    checkpoint=dict(save_best='coco/AP', rule='greater', max_keep_ckpts=1),
    logger=dict(interval=log_interval)
)
custom_hooks = [
    dict(type='EMAHook', ema_type='ExpMomentumEMA', momentum=0.0002, update_buffers=True, priority=49)
]

# evaluators
val_evaluator = dict(type='CocoMetric',
                     ann_file=data_root+'coco/annotations/person_keypoints_val2017.json')
test_evaluator = val_evaluator
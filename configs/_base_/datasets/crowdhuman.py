# dataset settings
dataset_type = 'CrowdHumanDataset'
data_root = 'data/CrowdHuman/'

img_norm_cfg = dict(
    mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='Resize', img_scale=(1400, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[[
            dict(
                type='Resize',
                img_scale=[(640, 1400), (672, 1400), (704, 1400),
                           (736, 1400), (768, 1400), (800, 1400),
                           (832, 1400), (864, 1400), (896, 1400)],
                multiscale_mode='value',
                keep_ratio=True)
        ],
                  [
                      dict(
                          type='Resize',
                          img_scale=[(400, 1400), (500, 1400), (600, 1400)],
                          multiscale_mode='value',
                          keep_ratio=True),
                      dict(
                          type='RandomCrop',
                          crop_type='absolute_range',
                          crop_size=(384, 600),
                          allow_negative_crop=True),
                      dict(
                          type='Resize',
                          img_scale=[(640, 1400), (672, 1400), (704, 1400),
                                     (736, 1400), (768, 1400), (800, 1400),
                                     (832, 1400), (864, 1400), (896, 1400)],
                          multiscale_mode='value',
                          override=True,
                          keep_ratio=True)
                  ]]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1400, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=('person', ),
        ann_file=data_root + 'annotations/train.json',
        img_prefix=data_root + 'Images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=('person', ),
        ann_file=data_root + 'annotations/val.json',
        img_prefix=data_root + 'Images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=('person', ),
        ann_file=data_root + 'annotations/val.json',
        img_prefix=data_root + 'Images/',
        pipeline=test_pipeline))
evaluation = dict(interval=5000,  # not evaluate during training.
                  metric='bbox',
                  classwise=True,
                  proposal_nums=(500, 500, 1000))

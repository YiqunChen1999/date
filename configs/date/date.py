# model settings

max_per_img = 100
model = dict(
    type='FCOS',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron/resnet50_caffe')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='DATEHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        deformable=False,
        predictors=[
            dict(
                type='OneToOneHeadPredictor',
                deformable=False,
                stop_grad=False,
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0),
                loss_box=dict(type='L1Loss', loss_weight=5.0),
                loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                assigner=dict(
                    type='POTOAssigner',
                    alpha=0.8,
                    iou_type='giou',
                    strides=[8, 16, 32, 64, 128],
                    center_sampling_radius=1.5,
                ),
                test_cfg=dict(
                    nms_pre=max_per_img,
                    min_bbox_size=0,
                    score_thr=0.01,
                    nms=False,
                    max_per_img=max_per_img)
            ),
            dict(
                type='FCOSHeadPredictor',
                center_sampling=False,
                center_sample_radius=1.5,
                centerness_on_reg=False,
                deformable=False,
                stop_grad=False,
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=1.0),
                loss_box=dict(type='IoULoss', loss_weight=1.0),
                loss_ctr=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=True,
                    loss_weight=1.0),
                train_cfg=dict(
                    assigner=dict(
                        type='MaxIoUAssigner',
                        pos_iou_thr=0.5,
                        neg_iou_thr=0.4,
                        min_pos_iou=0,
                        ignore_iof_thr=-1),
                    allowed_border=-1,
                    pos_weight=-1,
                    debug=False),
                test_cfg=dict(
                    nms_pre=1000,
                    min_bbox_size=0,
                    score_thr=0.05,
                    nms=dict(type='nms', iou_threshold=0.5),
                    max_per_img=100)
            )
        ],
        init_cfg=[
                dict(
                    type='Normal',
                    layer='Conv2d',
                    std=0.01,
                    override=dict(
                        type='Normal',
                        name='conv_cls',
                        std=0.01,
                        bias_prob=0.01)),
                dict(
                    type='Normal',
                    layer='ModulatedDeformableConv2d',
                    std=0.01,
                    override=dict(
                        type='Normal',
                        name='conv_cls',
                        std=0.01,
                        bias_prob=0.01))],
    )
)

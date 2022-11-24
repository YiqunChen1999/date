_base_ = './date_r50_12e_8x2.py'

max_per_img = 100
model = dict(
    bbox_head=dict(
        predictors=[
            dict(
                type='DeFCNPredictor',
                deformable=False,
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0),
                loss_box=dict(type='L1Loss', loss_weight=5.0),
                loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                loss_aux=None,  # The DeFCN without aux loss.
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
    )
)
optimizer = dict(
    type='AdamW',
    lr=4e-4,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))

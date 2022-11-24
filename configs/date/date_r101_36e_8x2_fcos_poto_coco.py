_base_ = './date_r50_36e_8x2.py'

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron/resnet101_caffe'))
)

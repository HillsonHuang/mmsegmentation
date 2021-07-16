_base_ = [
    '../_base_/models/fpn_r50.py', '../_base_/datasets/human_parsing.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]

model = dict(decode_head=dict(num_classes=20))
# model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))

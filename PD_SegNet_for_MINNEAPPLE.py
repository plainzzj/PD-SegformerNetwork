dataset_type = 'AppleDataset'
data_root = 'data/zzj_apple/MinneApple'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(720, 1080), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(512, 1024), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(512, 1024), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(720, 1280),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='AppleDataset',
        data_root='data/zzj_apple/MinneApple',
        img_dir='detection/train/images',
        ann_dir='detection/train/masks',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='Resize', img_scale=(720, 1080), ratio_range=(0.5, 2.0)),
            dict(type='RandomCrop', crop_size=(512, 1024), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(512, 1024), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type='AppleDataset',
        data_root='data/zzj_apple/MinneApple',
        img_dir='test_data/segmentation/images',
        ann_dir='test_data/segmentation/masks',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(720, 1280),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='AppleDataset',
        data_root='data/zzj_apple/MinneApple',
        img_dir='test_data/segmentation/images',
        ann_dir='test_data/segmentation/masks',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(720, 1280),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[60000, 72000],
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=1000)
evaluation = dict(interval=500, metric='All', pre_eval=True)
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b2_20220624-66e8bf70.pth'
norm_cfg = dict(type='SyncBN', requires_grad=True)
conv_kernel_size = 1
num_stages = 3
model = dict(
    type='CascadeEncoderDecoder',
    num_stages=2,
    pretrained=
    'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b2_20220624-66e8bf70.pth',
    backbone=dict(
        type='MixVisionTransformer',
        in_channels=3,
        embed_dims=64,
        num_stages=4,
        num_layers=[3, 4, 6, 3],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1),
    decode_head=[
        dict(
            type='IterativeDecodeHead',
            num_stages=3,
            kernel_update_head=[
                dict(
                    type='KernelUpdateHead',
                    num_classes=2,
                    num_ffn_fcs=2,
                    num_heads=8,
                    num_mask_fcs=1,
                    feedforward_channels=2048,
                    in_channels=256,
                    out_channels=256,
                    dropout=0.0,
                    conv_kernel_size=1,
                    ffn_act_cfg=dict(type='ReLU', inplace=True),
                    with_ffn=True,
                    feat_transform_cfg=dict(
                        conv_cfg=dict(type='Conv2d'), act_cfg=None),
                    kernel_updator_cfg=dict(
                        type='KernelUpdator',
                        in_channels=256,
                        feat_channels=256,
                        out_channels=256,
                        act_cfg=dict(type='ReLU', inplace=True),
                        norm_cfg=dict(type='LN'))),
                dict(
                    type='KernelUpdateHead',
                    num_classes=2,
                    num_ffn_fcs=2,
                    num_heads=8,
                    num_mask_fcs=1,
                    feedforward_channels=2048,
                    in_channels=256,
                    out_channels=256,
                    dropout=0.0,
                    conv_kernel_size=1,
                    ffn_act_cfg=dict(type='ReLU', inplace=True),
                    with_ffn=True,
                    feat_transform_cfg=dict(
                        conv_cfg=dict(type='Conv2d'), act_cfg=None),
                    kernel_updator_cfg=dict(
                        type='KernelUpdator',
                        in_channels=256,
                        feat_channels=256,
                        out_channels=256,
                        act_cfg=dict(type='ReLU', inplace=True),
                        norm_cfg=dict(type='LN'))),
                dict(
                    type='KernelUpdateHead',
                    num_classes=2,
                    num_ffn_fcs=2,
                    num_heads=8,
                    num_mask_fcs=1,
                    feedforward_channels=2048,
                    in_channels=256,
                    out_channels=256,
                    dropout=0.0,
                    conv_kernel_size=1,
                    ffn_act_cfg=dict(type='ReLU', inplace=True),
                    with_ffn=True,
                    feat_transform_cfg=dict(
                        conv_cfg=dict(type='Conv2d'), act_cfg=None),
                    kernel_updator_cfg=dict(
                        type='KernelUpdator',
                        in_channels=256,
                        feat_channels=256,
                        out_channels=256,
                        act_cfg=dict(type='ReLU', inplace=True),
                        norm_cfg=dict(type='LN')))
            ],
            kernel_generate_head=dict(
                type='SegformerHead',
                in_channels=[64, 128, 320, 512],
                in_index=[0, 1, 2, 3],
                channels=256,
                dropout_ratio=0.1,
                num_classes=2,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                align_corners=False,
                loss_decode=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0))),
        dict(
            type='ZZJPointHead',
            in_channels=[256, 256, 256, 256],
            in_index=[0, 1, 2, 3],
            channels=256,
            num_fcs=3,
            coarse_pred_each_layer=True,
            dropout_ratio=-1,
            num_classes=2,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
    ],
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=320,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    train_cfg=dict(
        num_points=2048, oversample_ratio=3, importance_sample_ratio=0.75),
    test_cfg=dict(
        mode='whole',
        subdivision_steps=2,
        subdivision_num_points=8196,
        scale_factor=2))
work_dir = './work_dirs/zzj_zzj_mitb2_apple_ours'
gpu_ids = range(0, 4)
auto_resume = False

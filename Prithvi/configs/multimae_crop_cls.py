
import os

dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None
resume_from = None
cudnn_benchmark = True
num_frames = 1
img_size = 224
num_workers = 2
custom_imports = dict(imports=["geospatial_fm"])

# model
# TO BE DEFINED BY USER: model path
pretrained_weights_path = "/mnt/isilon/space_cluster/multimae/checkpoint-vit-79.pth"
num_layers = 12
patch_size = 16
embed_dim = 768
num_heads = 8

max_epochs = 80
eval_epoch_interval = 5

loss_weights_multi = [
    0.386375,
    0.661126,
    0.548184,
    0.640482,
    0.876862,
    0.925186,
    3.249462,
    1.542289,
    2.175141,
    2.272419,
    3.062762,
    3.626097,
    1.198702,
]
loss_func = dict(
    type="CrossEntropyLoss",
    use_sigmoid=False,
    class_weight=loss_weights_multi,
    avg_non_ignore=True,
)
output_embed_dim = embed_dim


# TO BE DEFINED BY USER: Save directory
experiment = "multimae_crop_class"
project_dir = "/mnt/isilon/maloulou/multimae/semseg/"
work_dir = os.path.join(project_dir, experiment)
save_path = work_dir


dataset_type = "GeospatialDataset"

# TO BE DEFINED BY USER: data directory
data_root = "/mnt/isilon/space_cluster/multi-temporal-crop-classification/"

img_norm_cfg = dict(
    means=[
        494.905781,
        815.239594,
        924.335066,
        2968.881459,
        2634.621962,
        1739.579917,
        494.905781,
        815.239594,
        924.335066,
        2968.881459,
        2634.621962,
        1739.579917,
        494.905781,
        815.239594,
        924.335066,
        2968.881459,
        2634.621962,
        1739.579917,
    ],
    stds=[
        284.925432,
        357.84876,
        575.566823,
        896.601013,
        951.900334,
        921.407808,
        284.925432,
        357.84876,
        575.566823,
        896.601013,
        951.900334,
        921.407808,
        284.925432,
        357.84876,
        575.566823,
        896.601013,
        951.900334,
        921.407808,
    ],
)

bands = [3, 2, 1]

tile_size = 224
orig_nsize = 512
crop_size = (tile_size, tile_size)

train_pipeline = [
    dict(type="LoadGeospatialImageFromFile", to_float32=True),
    dict(type="LoadGeospatialAnnotations", reduce_zero_label=True),
    dict(type="RandomFlip", prob=0.5),
    dict(type="ToTensor", keys=["img", "gt_semantic_seg"]),
    # to channels first
    dict(type="TorchPermute", keys=["img"], order=(2, 0, 1)),
    dict(type="TorchNormalize", **img_norm_cfg),
    dict(type="TorchRandomCrop", crop_size=crop_size),
    dict(
        type="Reshape",
        keys=["img"],
        new_shape=(tile_size, tile_size, len(bands)*6,),
    ),
    dict(type='BandsExtract', bands=bands),
    dict(
        type="Reshape",
        keys=["img"],
        new_shape=(len(bands), tile_size, tile_size),
    ),
    dict(type="Reshape", keys=["gt_semantic_seg"], new_shape=(1, tile_size, tile_size)),
    dict(type="CastTensor", keys=["gt_semantic_seg"], new_type="torch.LongTensor"),
    dict(type="Collect", keys=["img", "gt_semantic_seg"]),
]

test_pipeline = [
    dict(type="LoadGeospatialImageFromFile", to_float32=True),
    dict(type="ToTensor", keys=["img"]),
    
    # to channels first
    dict(type="TorchPermute", keys=["img"], order=(2, 0, 1)),
    dict(type="TorchNormalize", **img_norm_cfg),
    dict(
        type="Reshape",
        keys=["img"],
        new_shape=(len(bands), -1, -1),
        look_up=dict({"2": 1, "3": 2}),
    ),
    dict(type="CastTensor", keys=["img"], new_type="torch.FloatTensor"),
    dict(
        type="CollectTestList",
        keys=["img"],
        meta_keys=[
            "img_info",
            "seg_fields",
            "img_prefix",
            "seg_prefix",
            "filename",
            "ori_filename",
            "img",
            "img_shape",
            "ori_shape",
            "pad_shape",
            "scale_factor",
            "img_norm_cfg",
        ],
    ),
]

CLASSES = (
    "Natural Vegetation",
    "Forest",
    "Corn",
    "Soybeans",
    "Wetlands",
    "Developed/Barren",
    "Open Water",
    "Winter Wheat",
    "Alfalfa",
    "Fallow/Idle Cropland",
    "Cotton",
    "Sorghum",
    "Other",
)

dataset = "GeospatialDataset"

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
    train=dict(
        type=dataset,
        CLASSES=CLASSES,
        reduce_zero_label=True,
        data_root=data_root,
        img_dir="training_chips",
        ann_dir="training_chips",
        pipeline=train_pipeline,
        img_suffix="_merged.tif",
        seg_map_suffix=".mask.tif",
    ),
    val=dict(
        type=dataset,
        CLASSES=CLASSES,
        reduce_zero_label=True,
        data_root=data_root,
        img_dir="validation_chips",
        ann_dir="validation_chips",
        pipeline=test_pipeline,
        img_suffix="_merged.tif",
        seg_map_suffix=".mask.tif",
    ),
    test=dict(
        type=dataset,
        CLASSES=CLASSES,
        reduce_zero_label=True,
        data_root=data_root,
        img_dir="validation_chips",
        ann_dir="validation_chips",
        pipeline=test_pipeline,
        img_suffix="_merged.tif",
        seg_map_suffix=".mask.tif",
    ),
)

optimizer = dict(type="Adam", lr=1.5e-05, betas=(0.9, 0.999), weight_decay=0.05)
optimizer_config = dict(grad_clip=None)

lr_config = dict(
    policy="poly",
    warmup="linear",
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False,
)
"""dict(
            type='MMSegWandbHook',
            by_epoch=False,
            interval=1,
            with_step=False,
            init_kwargs=dict(
                project='multimae-sem-seg',
                name='multi-temporal-crop-classification-multimae'),
            log_checkpoint=True,
            log_checkpoint_metadata=True,
            num_eval_images=10)"""
log_config = dict(
    interval=1,
    hooks=[dict(type='TextLoggerHook', by_epoch=False)]
        )

checkpoint_config = dict(by_epoch=True, interval=100, out_dir=save_path)

evaluation = dict(
    interval=eval_epoch_interval,
    metric="mIoU",
    pre_eval=True,
    save_best="mIoU",
    by_epoch=True,
)
reduce_train_set = dict(reduce_train_set=False)
reduce_factor = dict(reduce_factor=1)
runner = dict(type="EpochBasedRunner", max_epochs=max_epochs)
workflow = [("train", 1)]
norm_cfg = dict(type="BN", requires_grad=True)

'''neck=dict(
        type="ConvTransformerTokensToEmbeddingNeck",
         embed_dim=embed_dim,
         output_embed_dim=output_embed_dim,
         drop_cls_token=True,
         Hp=16,
         Wp=16,
)'''

model = dict(
    type="EncoderDecoder",
    backbone=dict(
        type="VisionTransformer",
        pretrained=pretrained_weights_path,
        img_size=img_size,
        norm_cfg = dict(type='LN', eps=1e-6, elementwise_affine=True),
        act_cfg = dict(type='GELU'),
        patch_size=patch_size,
        num_layers=12,
        mlp_ratio=4,
        num_heads=1,
        in_channels=1,
           
    ),
    decode_head=dict(
        num_classes=len(CLASSES),
        in_channels=output_embed_dim,
        type="FCNHead",
        in_index=-1,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        norm_cfg=dict(type="BN", requires_grad=True),
        align_corners=False,
        loss_decode=loss_func,
    ),
    train_cfg=dict(),
    test_cfg=dict(
        mode="slide",
        stride=(int(tile_size / 2), int(tile_size / 2)),
        crop_size=(tile_size, tile_size),
    ),
)
auto_resume = False

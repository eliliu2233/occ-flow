_base_ = [
    '../_base_/dataset_v1.py',
    '../_base_/optimizer.py',
    '../_base_/schedule.py',
]

queue_length = 3
find_unused_parameters=True
data_config = dict(image_size=[900,1600], input_size=[256,704], render_size=[512,1408])
resize_config = dict(input_scale=0.44, render_scale=0.88)
num_rays = [32, 88]
amp = False
# flow = True
tracking = True
flow_epoch = 0
tracking_flow_path = 'data/nuscenes/nuscenes_flow'
# flow_ckpt = 'ckpts/gmflow-scale2-regrefine6-nuscenes-150k.pth'
print_freq = 10
eval_every_epochs = 2
max_epochs = 12
num_gpus = 6
num_iters = int(max_epochs * (3516*8) / num_gpus)
# num_iters = int(max_epochs * (6019) / num_gpus)
warmup_iters = 1000
label_path = {
    'semantic':'data/nuscenes_semantic',
}
cam_types = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'] #

multisteplr = True
multisteplr_config = dict(
    decay_t = [int(0.6*num_iters), int(0.8*num_iters)],
    decay_rate = 0.2,
    warmup_t = warmup_iters,
    warmup_lr_init = 1e-6,
    t_in_epochs = False
)

optimizer = dict(
    optimizer=dict(
        type='AdamW',
        lr=1e-4,
        weight_decay=0.01,
        # eps=1e-4
    ),
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
            }
    ),
)


data_path = 'data/nuscenes/'

train_dataset_config = dict(
    _delete_=True,
    type='nuScenes_Tracking_Dataset',
    data_path = data_path,
    queue_length=queue_length,
    imageset = 'data/nuscenes_sweeps_infos_train_anns.pkl',
    crop_size = data_config['render_size'],
    resize_scale = resize_config['render_scale'],
    sensor_type = cam_types,
    # aux_sync = True,
    input_img_crop_size = data_config['input_size'],
    min_dist = 0.4,
    max_dist = 6.0,
    strict = True,
    return_depth = False,
    eval_depth = 80,
    cur_prob = 1.0,
    prev_prob = 0.5,
    choose_nearest = True,
    ref_sensor = 'CAM_FRONT',
    composite_prev_next=True,
    sensor_mus=[0.5, 0.5],
    sensor_sigma=0.5,
    ego_centric=True,
    # debug=True,
    # with_nearest_next=True
)
    
val_dataset_config = dict(
    _delete_=True,
    queue_length=queue_length,
    type='nuScenes_Tracking_Dataset',
    data_path = data_path,
    imageset = 'data/nuscenes_sweeps_infos_val_anns.pkl',
    crop_size = data_config['render_size'],
    resize_scale = resize_config['render_scale'],
    sensor_type = cam_types,
    aux_sync = True,
    input_img_crop_size = data_config['input_size'],
    min_dist = 0.0,
    max_dist = 6.0,
    strict = False,
    return_depth = True,
    eval_depth = 80,
    cur_prob = 1,
    # step=2,
    prev_prob = 0.5,
    choose_nearest = True,
    ref_sensor = 'CAM_FRONT',
    composite_prev_next=True,
    sensor_mus=[0.5, 0.5],
    sensor_sigma=0.5,
    ego_centric=True,
    with_nearest_next=True,
    # select_scene=['scene-0797'],
    label_path=label_path,
)

train_wrapper_config = dict(
    type='tpvformer_dataset_nuscenes_temporal',
    phase='train', 
    mode=2,
    data_config=data_config, resize_config=resize_config,
    scale_rate=0.5,
    photometric_aug=dict(
        use_swap_channel=False,
    ),
    # pad_img_size=[416, 864],
    # random_scale=[0.92, 1.08],
    # pad_scale_rate=[0.5417, 0.54],
    # use_flip=True
)

val_wrapper_config = dict(
    type='tpvformer_dataset_nuscenes_temporal',
    phase='val', 
    mode=2,
    data_config=data_config, resize_config=resize_config,
    scale_rate=0.5,
    photometric_aug=dict(
        use_swap_channel=False,
    ),
    # pad_img_size=[416, 864],
    # pad_scale_rate=[0.5417, 0.54],
)

train_loader = dict(
    batch_size = 1,
    shuffle = True,
    num_workers = 1,
)
    
val_loader = dict(
    batch_size = 1,
    shuffle = False,
    num_workers = 1,
)

loss = dict(
    type='MultiLoss',
    loss_cfgs=[
        dict(
            type='ReprojLossMonoMultiNewCombine',
            weight=1.0,
            no_ssim=False,
            img_size=data_config['render_size'],
            ray_resize=num_rays,
            input_dict={
                'curr_imgs': 'curr_imgs',
                'prev_imgs': 'prev_imgs',
                'next_imgs': 'next_imgs',
                'ray_indices': 'ray_indices',
                'weights': 'weights',
                'ts': 'ts',
                'metas': 'metas',
                'ms_rays': 'ms_rays',
                # 'deltas': 'deltas'
                }),
        dict(
            type='EikonalLoss',
            weight=0.1,),
        dict(
            type='LidarLoss',
            weight=0.2,
            silog=True,
            discard_outliers=0.0),
        dict(
            type='SecondGradLoss',
            weight=0.05),
        dict(
            type='EdgeLoss3DMS',
            weight=0.02,
            img_size=data_config['render_size'],
            ray_resize=num_rays,
        ),
        ],
    count_freq=print_freq)

loss_input_convertion = dict(
    ms_depths='ms_depths',
    ms_rays='ms_rays',
    # ms_accs='ms_accs',
    # ms_colors='ms_colors',
    ray_indices='ray_indices',
    weights='weights',
    ts='ts',
    eik_grad='eik_grad',
    second_grad='second_grad',
    uniform_sdf='uniform_sdf',
    pred_ranges='pred_ranges',
)

load_from = ''

_dim_ = 192
_ffn_dim_ = 2 * _dim_
num_heads = 6
mapping_args = dict(
    nonlinear_mode='linear',
    h_size=[100, 0],
    h_range=[40.0, 0],
    h_half=False,
    w_size=[100, 0],
    w_range=[40.0, 0],
    w_half=False,
    d_size=[20, 0],
    d_range=[-1.0, 5.4, 5.4]
    # d_range=[-3.0, 3.4, 3.4]
)
tpv_mapping_args = dict(
    nonlinear_mode='linear',
    h_size=[50, 0],
    h_range=[40.0, 0],
    h_half=False,
    w_size=[50, 0],
    w_range=[40.0, 0],
    w_half=False,
    d_size=[10, 0],
    d_range=[-1.0, 5.4, 5.4]
    # d_range=[-3.0, 3.4, 3.4]
)
tpv_h = 100
tpv_w = 100
tpv_z = 10
point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
# point_cloud_range = [-40.0, -40.0, -3.0, 40.0, 40.0, 3.4]

num_points_cross = [48, 48, 8]
num_points_self = 12


self_cross_layer = dict(
    type='TPVFormerLayer',
    attn_cfgs=[
        dict(
            type='CrossViewHybridAttention',
            embed_dims=_dim_,
            num_heads=num_heads,
            num_levels=3,
            num_points=num_points_self,
            dropout=0.1,
            batch_first=True),
        dict(
            type='TPVCrossAttention',
            embed_dims=_dim_,
            num_cams=len(cam_types),
            dropout=0.1,
            batch_first=True,
            num_heads=num_heads,
            num_levels=4,
            num_points=num_points_cross)
    ],
    feedforward_channels=_ffn_dim_,
    ffn_dropout=0.1,
    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
)


model = dict(
    type='TPVSegmentor',
    queue_length=queue_length,
    is_corner=False,
    volume_size=[tpv_h,tpv_w,tpv_z], 
    img_backbone_out_indices=[0, 1, 2, 3],
    img_backbone=dict(
        type="ConvNeXt",
        arch="base",
        out_indices=(0, 1, 2, 3),
        norm_out=True,
        frozen_stages=0,
        init_cfg=dict(
            type="Pretrained",
            checkpoint="ckpts/convnextB_1kpretrained_official_style.pth",
        ),
    ),
    img_neck=dict(
        type='FPN',
        # in_channels=[256, 512, 1024, 2048],
        in_channels=[128, 256, 512, 1024],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True),
    lifter=dict(
        type='TPVQueryLifter',
        tpv_h=tpv_h,
        tpv_w=tpv_w,
        tpv_z=tpv_z, 
        dim=_dim_),
    encoder=dict(
        type='TPVFormerEncoder',
        mapping_args=tpv_mapping_args,
        is_corner=False,

        embed_dims=_dim_,
        num_cams=len(cam_types),
        num_feature_levels=4,
        positional_encoding=dict(
            type='TPVPositionalEncoding',
            num_freqs=[12] * 3, 
            embed_dims=_dim_, 
            tot_range=point_cloud_range),
        num_points_cross=num_points_cross,
        num_points_self=[num_points_self] * 3,
        transformerlayers=[
            self_cross_layer,
            self_cross_layer,
            self_cross_layer,
            self_cross_layer], 
        num_layers=4),
    temporal_fusion=dict(
        type='AFFM',
        attention=dict(
            type='TemporalBEVAttention',
            embed_dims=_dim_,
            num_heads=num_heads,
            num_levels=1,
            dropout=0.0,
            num_points=8),
        bev_h=tpv_h, bev_w=tpv_w, num_queue=queue_length,
    ),
    temporal_encoder_backbone=dict(
        type='CustomResNet3D',
        # numC_input=_dim_ * queue_length,
        numC_input=_dim_,
        num_layer=[1, 2, 4],
        with_cp=False,
        num_channels=[_dim_,_dim_*2,_dim_*4],
        stride=[1,2,2],
        backbone_output_ids=[0,1,2]),
    temporal_encoder_neck=dict(
        type='FPN3D',
        with_cp=False,
        in_channels=[_dim_,_dim_*2,_dim_*4],
        out_channels=_dim_,
        norm_cfg=dict(type='BN3d', requires_grad=True),
    ),
    decoder=dict(
        type='Occ_Decoder',
        with_cp=False,
        norm_cfg=dict(type='BN3d', requires_grad=True),
        soft_weights=True,
        num_level=3,
        in_channels=[_dim_]*3,
    ),
    head=dict(
        type='NeuSHead',
        roi_aabb=point_cloud_range, 
        resolution=0.4,
        near_plane=0.0,
        far_plane=1e10,
        num_samples=256,
        num_samples_importance=0,
        num_up_sample_steps=0,
        base_variance=4,

        beta_init=0.1,
        beta_max=0.195,
        total_iters=3516*11,
        beta_hand_tune=False,
        # beta_learnable=False,
        
        use_numerical_gradients=False,
        sample_gradient=True,
        use_compact_2nd_grad=True,
        return_uniform_sdf=True,
        return_second_grad=True,
        # return_uniform_3d_flow=True,

        # rays args
        ray_sample_mode='cellular',    # fixed, cellular
        ray_number=num_rays,      # 192 * 400
        ray_img_size=data_config['render_size'],
        ray_upper_crop=0,
        # img2lidar args
        trans_kw='temImg2lidar',
        novel_view=None,

        # render args
        render_bkgd='random',

        mapping_args=mapping_args,
        # estimate_flow_my=True,

        # mlp decoder 
        embed_dims=int(_dim_/2),
        color_dims=0,
        density_layers=2,
        sh_deg=0,
        sh_act='relu',
        two_split=False,
        representation='tpv'))
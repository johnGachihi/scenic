import ml_collections

VARIANT = 'S/16'
TRAIN_SIZE = 284


SENTINEL2_L2A_MEAN = [1431, 1233, 1209, 1192, 1448, 2238, 2609, 2537, 2828, 884, 2226, 1537]
SENTINEL2_L2A_STD = [157, 254, 290, 420, 363, 457, 575, 606, 630, 156, 554, 523]

def get_config():
    config = ml_collections.ConfigDict()
    config.experiment_name = '100ep_run'
    config.dataset_name = 'finetuning_dataset'
    config.data_dtype_str = 'float32'
    config.dataset_configs = ml_collections.ConfigDict()
    input_resolution = 224

    # Data
    config.dataset_configs.dataset = 'sen1_floods11'  # sen1_floods11, substation
    config.dataset_configs.train_split = 'train'
    config.dataset_configs.val_split = 'val'
    config.dataset_configs.test_split = 'test'
    config.dataset_configs.num_classes = 2
    bands = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]
    config.dataset_configs.input_shape = (input_resolution, input_resolution, len(bands))

    config.dataset_configs.pp_train = (
        'permute_channels_last("s2_img")'
        '|permute_channels_last("s1_img")'
        '|permute_channels_last("label")'
        f'|select_bands({bands}, "s2_img",)'
        f'|resize({input_resolution}, data_key="s2_img")'
        f'|resize({input_resolution}, data_key="s1_img")'
        f'|resize({input_resolution}, "nearest", data_key="label")'
        '|concat("s2_img", "s1_img", "image", axis=-1)'
        f'|keep("image", "label")'
    )
    config.dataset_configs.pp_eval = (
        f'permute_channels_last("s2_img")'
        '|permute_channels_last("s1_img")'
        '|permute_channels_last("label")'
        f'|select_bands({bands}, "s2_img",)'
        f'|resize({input_resolution}, data_key="s2_img")'
        f'|resize({input_resolution}, data_key="s1_img")'
        f'|resize({input_resolution}, "nearest", data_key="label")'
        '|concat("s2_img", "s1_img", "image", axis=-1)'
        f'|keep("image", "label")'
    )

    config.dataset_configs.shuffle_buffer_size = 5  # TODO: tune

    # Model
    config.model = ml_collections.ConfigDict()
    ## Semantic Segmentation
    config.model.num_classes = 2  # TODO: Use dataset_configs.num_classes
    ## ViT-LOCA Encoder Model
    version, patch = VARIANT.split('/')
    patch = int(patch)
    config.model.hidden_size = {'Ti': 192,
                                'S': 384,
                                'B': 768,
                                'L': 1024,
                                'H': 1280}[version]
    config.model.patches = ml_collections.ConfigDict()
    config.model.patches.size = [patch, patch]
    config.model.mlp_dim = {'Ti': 768,
                            'S': 1536,
                            'B': 3072,
                            'L': 4096,
                            'H': 5120}[version]
    config.model.num_layers = {'Ti': 12,
                               'S': 12,
                               'B': 12,
                               'L': 24,
                               'H': 32}[version]
    config.model.num_heads = {'Ti': 3, 'S': 6, 'B': 12, 'L': 16, 'H': 16}[version]

    # Sentinel 2 channel-grouping
    config.sen2grouped = True
    # B1:Aerosol = 0, B2:Blue = 1, B3:Green = 2, B4:Red = 3,
    # B5:RedEdge1 = 4, B6:RedEdge2 = 5, B7:RedEdge3 = 6, B8:NIR = 7, B8A:RedEdge4 = 8,
    # B9:WaterVapor = 9, B11:SWIR1 = 10, B12:SWIR2 = 11
    config.sen2changroups = ((1, 2, 3, 7), (4, 5, 6, 8), (10, 11),
                             # (12,), (13,)
                             )

    # Multimodal
    # config.multimodal = 'early_concat_s2_and_s1'

    # LOCA specific parameters
    config.n_ref_positions = int((input_resolution // patch) ** 2)
    config.reference_seqlen = config.n_ref_positions
    config.apply_cluster_loss = False  # Always false for finetuning

    # Training
    config.batch_size = 16
    config.eval_batch_size = 16
    config.num_training_epochs = 350
    config.rng_seed = 42
    steps_per_epoch = TRAIN_SIZE // config.batch_size
    total_steps = config.num_training_epochs * steps_per_epoch

    # config.pretrained_weights = '/home/admin/john/scenic/loca_mmearth64_small_16patches_224size_sen2grouped_2/checkpoint_655005'
    # config.pretrained_weights = '/home/admin/john/scenic/loca_mmearth64_300k_small_16patches_224size/checkpoint_937500'
    # config.pretrained_weights = '/home/admin/john/scenic/loca_mmearth64_small_16patches_224size_sen2grouped_maintain_seqlen/checkpoint_468700'
    # config.pretrained_weights =   '/home/admin/john/scenic/loca_mmearth64_small_16patches_224size_sen2grouped/checkpoint_655005'
    # config.pretrained_weights =   '/home/admin/john/scenic/loca_300k_56s_8p/checkpoint_234300'
    # config.pretrained_weights =   '/home/admin/john/scenic/loca_300k_56s_4p/checkpoint_234300'
    # config.pretrained_weights =   '/home/admin/john/scenic/loca_300k_56_4_early_fusion_sen1_to_sen2rgb'
    # config.pretrained_weights =   '/home/admin/john/scenic/loca_300k_224_51_early_concat/checkpoint_468700'
    # config.pretrained_weights = '/home/admin/john/scenic/loca_56_4_early_concat_early_group_sampling_ref/checkpoint_468700'
    # config.pretrained_weights = '/home/admin/john/scenic/loca_mmearth64_small_16patches_224size_sen2grouped_maintain_seqlen/checkpoint_468700'
    # config.pretrained_weights = '/home/admin/john/scenic/loca_mmearth64_small_16patches_224size_sen2grouped/checkpoint_937500'
    # config.pretrained_weights = '/home/admin/john/scenic/loca_same_group_masking_enc_and_cross/checkpoint_468700'
    # config.pretrained_weights = '/home/admin/john/scenic/loca_same_group_attn_masking_0_25/checkpoint_468700'
    # config.pretrained_weights = '/home/admin/john/scenic/loca_mmearth64_small_16patches_224size_sen2grouped/checkpoint_937500'
    # config.pretrained_weights = '/home/admin/satellite-loca/scenic/loca_same_group_mask_ref_and_query/checkpoint_468700'
    # config.pretrained_weights = '/home/admin/satellite-loca/scenic/loca_same_group_attn_masking_0_25/checkpoint_468700'
    # config.pretrained_weights = '/home/admin/john/scenic/loca_add_dem_early_fuse/checkpoint_468700'
    # config.pretrained_weights = '/home/admin/john/scenic/loca_add_dem_early_concat_with_group_sampling/checkpoint_468700'
    # config.pretrained_weights = '/home/admin/john/scenic/loca_s1_s2_dem_attn_masking_enc_cross_0_4_masking/checkpoint_468700'
    # config.pretrained_weights = '/home/admin/john/scenic/loca_s1_s2_dem_attn_masking_enc_cross_0_3_masking/checkpoint_468700'
    # config.pretrained_weights = '/home/admin/john/scenic/loca_s1_s2_dem_no_attn_masking_enc_cross_0_4_masking/checkpoint_468700'
    # config.pretrained_weights = '/home/admin/john/scenic/loca_s1_s2_dem_attn_masking_enc_cross_0_0_masking/checkpoint_468700'
    # config.pretrained_weights = '/home/admin/john/scenic/loca_s1_s2_dem_attn_masking_enc_cross_0_4_masking_correct_band_order/checkpoint_468700'
    # config.pretrained_weights = '/home/admin/john/scenic/loca_s1_s2_dem_no_attn_masking_enc_cross_0_0_masking/checkpoint_468700'
    # config.pretrained_weights = '/home/admin/john/scenic/loca_s1_s2_dem_attn_masking_enc_cross_0_4_masking_correct_band_order_full_s1/checkpoint_468700'
    # config.pretrained_weights = '/home/admin/john/scenic/loca_300k_224s_16p/checkpoint_937500'
    # config.pretrained_weights = '/home/admin/john/scenic/loca_s1_s2_dem_no_attn_masking_enc_cross_0_4_masking_correct_band_order_full_s1/checkpoint_468700'
    config.pretrained_weights = '/home/admin/john/scenic/loca_s1_s2_dem_no_attn_masking_enc_cross_0_0_masking_correct_band_order_full_s1/checkpoint_468700'

    # Learning rate.
    config.lr_configs = ml_collections.ConfigDict()
    config.lr_configs.learning_rate_schedule = 'compound'
    config.lr_configs.factors = 'constant * cosine_decay * linear_warmup'
    config.lr_configs.warmup_steps = steps_per_epoch * 5
    config.lr_configs.steps_per_cycle = total_steps
    config.lr_configs.base_learning_rate = 0.001 * config.batch_size / 1024
    config.lr_configs.alpha = 0.01

    # Weight decay.
    config.weight_decay = 0.1

    # Logging.
    config.write_summary = True
    config.xprof = True  # Profile using xprof.
    # config.checkpoint = True  # Do checkpointing.
    # config.checkpoint_steps = 10000
    config.log_summary_steps = steps_per_epoch
    config.log_eval_steps = steps_per_epoch

    return config
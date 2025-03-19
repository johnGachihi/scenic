import ml_collections

VARIANT = 'S/16'
_SEN1_FLOODS11_TRAIN_SIZE = 284
_SUBSTATION_TRAIN_SIZE = 26522 * 0.8

SENTINEL2_L2A_MEAN = [1349.3977794889083, 1479.9521800379623, 1720.3688077425966, 1899.1848715975957,
                      2253.9309600057886, 2808.2001963620132, 3003.424149045887, 3149.5364927329806,
                      3110.840562275062, 3213.7636154015954, 2399.086213373806,
                      1811.7986415136786]
SENTINEL2_L2A_STD = [2340.2916479338087, 2375.872101251672, 2256.8997709659416, 2354.181051828758,
                     2292.99569489449, 2033.2166835293804, 1920.1736418230105, 1816.6152354201365,
                     1988.1938283738782, 1947.9031620588928, 1473.224812450967,
                     1390.6781165633136]

def get_config():
    config = ml_collections.ConfigDict()
    config.experiment_name = '100ep_run'
    config.dataset_name = 'finetuning_dataset'
    config.data_dtype_str = 'float32'
    config.dataset_configs = ml_collections.ConfigDict()
    input_resolution = 224

    # Data
    config.dataset_configs.dataset = 'substation'  # sen1_floods11, substation
    config.dataset_configs.train_split = 'train'
    config.dataset_configs.val_split = 'val'
    config.dataset_configs.test_split = 'test'
    config.dataset_configs.num_classes = 2
    bands = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]
    config.dataset_configs.input_shape = (input_resolution, input_resolution, len(bands))

    if config.dataset_configs.dataset == 'sen1_floods11':
        config.dataset_configs.pp_train = (
            f'permute_channels_last("s2_img")'
            '|permute_channels_last("label")'
            f'|select_bands({bands}, "s2_img",)'
            f'|resize({input_resolution}, data_key="s2_img")'
            f'|resize({input_resolution}, "nearest", data_key="label")'
            f'|keep("s2_img", "label")'
        )
        config.dataset_configs.pp_eval = (
            f'permute_channels_last("s2_img")'
            '|permute_channels_last("label")'
            f'|select_bands({bands}, "s2_img",)'
            f'|resize({input_resolution}, data_key="s2_img")'
            f'|resize({input_resolution}, "nearest", data_key="label")'
            f'|keep("s2_img", "label")'
        )
    else:
        config.dataset_configs.pp_train = (
            f'permute_channels_last("input")'
            f'|select_bands({bands}, "input",)'
            f'|standardize(mean={SENTINEL2_L2A_MEAN}, std={SENTINEL2_L2A_STD}, data_key="input")'
            f'|resize({input_resolution}, data_key="input")'
            f'|unsqueeze("label", -1)'
            f'|resize({input_resolution}, "nearest", data_key="label")'
            f'|cast("int32", "label")'
            f'|keep("input", "label")'
        )
        config.dataset_configs.pp_eval = (
            f'permute_channels_last("input")'
            f'|select_bands({bands}, "input",)'
            f'|standardize(mean={SENTINEL2_L2A_MEAN}, std={SENTINEL2_L2A_STD}, data_key="input")'
            f'|resize({input_resolution}, data_key="input")'
            f'|unsqueeze("label", -1)'
            f'|resize({input_resolution}, "nearest", data_key="label")'
            f'|cast("int32", "label")'
            f'|keep("input", "label")'
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

    # LOCA specific parameters
    config.n_ref_positions = int((input_resolution // patch) ** 2)
    config.reference_seqlen = config.n_ref_positions
    config.apply_cluster_loss = False  # Always false for finetuning

    # Training
    config.batch_size = 16
    config.eval_batch_size = 16
    config.num_training_epochs = 100
    config.rng_seed = 42
    steps_per_epoch = (_SEN1_FLOODS11_TRAIN_SIZE if config.dataset_configs.dataset == 'sen1_floods11' else _SUBSTATION_TRAIN_SIZE) // config.batch_size
    total_steps = config.num_training_epochs * steps_per_epoch

    # config.pretrained_weights = '/home/admin/satellite-loca/scenic/loca_mmearth64_small_32patches_224size/checkpoint_78100'

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
    config.log_summary_steps = 100
    config.log_eval_steps = 1

    return config
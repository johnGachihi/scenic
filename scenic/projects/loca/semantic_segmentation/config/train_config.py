import ml_collections

VARIANT = 'S/16'
_SEN1_FLOODS11_TRAIN_SIZE = 284

def get_config():
    config = ml_collections.ConfigDict()
    config.experiment_name = '100ep_run'
    config.dataset_name = 'finetuning_dataset'
    config.data_dtype_str = 'float32'
    config.dataset_configs = ml_collections.ConfigDict()
    input_resolution = 224

    # Data
    config.dataset_configs.pp_train = (
        'permute_channels_last("s2_img")'
        '|permute_channels_last("label")'
        '|resize(224, data_key="s2_img")'
        '|resize(224, "nearest", data_key="label")'
        '|keep("s2_img", "label")'
    )
    config.dataset_configs.shuffle_buffer_size = 25000

    # Sen1Floods11 Dataset
    config.dataset_configs.dataset = 'sen1_floods11'
    config.dataset_configs.train_split = 'train'

    # Model
    config.model = ml_collections.ConfigDict()
    ## Semantic Segmentation
    config.model.num_classes = 2
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
    config.n_ref_positions = int((input_resolution // patch)**2)
    config.reference_seqlen = config.n_ref_positions
    config.apply_cluster_loss = False  # Always false for finetuning

    # Training
    config.batch_size = 64
    config.num_training_epochs = 100
    config.rng_seed = 42
    steps_per_epoch = _SEN1_FLOODS11_TRAIN_SIZE // config.batch_size
    total_steps = config.num_training_epochs * steps_per_epoch

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
    config.log_summary_steps = 50

    return config
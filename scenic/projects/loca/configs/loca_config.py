# Copyright 2024 The Scenic Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=line-too-long
"""Default config for LOCA training on ImageNet2012 for 100 epochs."""

import ml_collections

VARIANT = 'S/4'
_MMEARTH_TRAIN_SIZE = 300_000

"""
Sentinel2_l2a": {
"sentinel2_l1c": {"mean": , "std": [1520.0684839687428, 1575.4239525583005, 1474.3747757041376, 1734.9206729983894, 1697.1412804437439, 1584.959816138674, 1577.9910344404889, 1560.2251591506092, 1519.2164490452863, 823.3855623314192, 61.70737973208095, 1311.5885770761618, 1140.1057025823181], "min": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], "max": [28000.0, 23163.0, 25441.0, 28002.0, 28002.0, 28003.0, 28004.0, 28004.0, 28003.0, 28000.0, 4722.0, 15007.0, 18675.0]},
"""

SENTINEL2_L1C_MEAN = [1864.880176877408, 1656.9923384425733, 1561.2433627865414, 1633.6041005007844,
                      1846.642924880421, 2422.1354550099322, 2706.1684786306714, 2907.509651871235,
                      2620.484567631748, 974.4786592695265, 2154.0573745085508,
                      1464.8020890030184]
SENTINEL2_L1C_STD = [1520.0684839687428, 1575.4239525583005, 1474.3747757041376, 1734.9206729983894,
                     1697.1412804437439, 1584.959816138674, 1577.9910344404889, 1560.2251591506092,
                     1519.2164490452863, 823.3855623314192, 1311.5885770761618,
                     1140.1057025823181]
SENTINEL2_L2A_MEAN = [1349.3977794889083, 1479.9521800379623, 1720.3688077425966, 1899.1848715975957,
                      2253.9309600057886, 2808.2001963620132, 3003.424149045887, 3149.5364927329806,
                      3110.840562275062, 3213.7636154015954, 2399.086213373806,
                      1811.7986415136786]
SENTINEL2_L2A_STD = [2340.2916479338087, 2375.872101251672, 2256.8997709659416, 2354.181051828758,
                     2292.99569489449, 2033.2166835293804, 1920.1736418230105, 1816.6152354201365,
                     1988.1938283738782, 1947.9031620588928, 1473.224812450967,
                     1390.6781165633136]

SENTINEL1_MEAN = [-11.673448681503809, -19.04684937464606, -10.525772696311732, -19.532663009778172,
                  -11.754322832475555, -19.187027531976366, -11.729595910348051, -21.294408727496148]
SENTINEL1_STD = [5.1287824026286755, 6.432180454976982, 5.646047662186715, 7.724311760316525,
                 5.01231147683192, 6.296322894653895, 5.348371323330351, 7.184819430273331]


def get_config():
  """Returns the default config for a 100 epoch LOCA training on ImageNet2012."""

  version, patch = VARIANT.split('/')
  patch = int(patch)

  config = ml_collections.ConfigDict()
  config.experiment_name = '100ep_run'
  # Dataset.
  config.dataset_name = 'loca_dataset'
  config.data_dtype_str = 'float32'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.prefetch_to_device = 1
  config.dataset_configs.shuffle_buffer_size = 25_000
  reference_resolution = 56
  reference_patch_width = reference_resolution // patch
  query_rand_res = reference_resolution
  query_rand_mask_res = query_rand_res // patch  # Should be equal to patch width/height of rand query
  query_foc_res = 24
  query_foc_mask_res = query_foc_res // patch  # Should be equal to patch width/height of focal query
  n_queries = 10
  config.dataset_configs.number_of_focal_queries = n_queries - 1
  config.dataset_configs.pp_train = (
      # Sentinel2 preprocessing.
      'permute_channels_last("sentinel2")' +
      f'|standardize_sentinel2({SENTINEL2_L1C_MEAN}, {SENTINEL2_L1C_STD}, {SENTINEL2_L2A_MEAN}, {SENTINEL2_L2A_STD}, "sentinel2", "sentinel2_type")' +

      # Sentinel1 preprocessing
      '|permute_channels_last("sentinel1")' +
      f'|standardize({SENTINEL1_MEAN}, {SENTINEL1_STD}, data_key="sentinel1")' +

      # Concatenate Sentinel1 and Sentinel2 along channel dimension
      '|concat("sentinel1", "sentinel2", "sentinel1_2", axis=-1)' +

      '|copy("sentinel1_2", "reference")' +
      f'|init_patch_matching_tracker({reference_patch_width}, "target_mask")' +
      '|init_box_tracker("target_box")' +
      f'|cropflip_generatemask({reference_resolution}, 32, flip=False, inkey=("reference", "target_mask", "target_box"), outkey=("reference", "target_mask", "target_box"))' +
      # '|value_range(0, 1, data_key="reference")' +
      # '|random_color_jitter(0.8, 0.4, 0.4, 0.2, 0.1, data_key="reference")' +
      # '|random_grayscale(0.2, data_key="reference")' +
      # '|random_blur(1.0, data_key="reference")' +
      # f'|standardize({MEAN_RGB}, {STDDEV_RGB}, data_key="reference")' +
      ''.join([f'|copy("sentinel1_2", "query{i}")' for i in range(n_queries)]) +
      f'|inception_crop_with_mask(({query_rand_res}, {query_rand_res}), 32, 100, ({query_rand_mask_res}, {query_rand_mask_res}), inkey=("query0", "target_mask", "target_box"), outkey=("query0", "query0_mask", "query0_box"))' +
      ''.join([
                f'|inception_crop_with_mask(({query_foc_res}, {query_foc_res}), 5, 32, ({query_foc_mask_res}, {query_foc_mask_res}), inkey=("query{i}", "target_mask", "target_box"), outkey=("query{i}", "query{i}_mask", "query{i}_box"))'
                for i in range(1, n_queries)]) +
      ''.join([f'|flip_with_mask(inkey=("query{i}", "query{i}_mask"), outkey=("query{i}", "query{i}_mask"))' for i in
               range(n_queries)]) +
      # ''.join([f'|value_range(0, 1, data_key="query{i}")' for i in range(n_queries)]) +
      # ''.join([f'|random_color_jitter(0.8, 0.4, 0.4, 0.2, 0.1, data_key="query{i}")' for i in range(n_queries)]) +
      # ''.join([f'|random_grayscale(0.2, data_key="query{i}")' for i in range(n_queries)]) +
      # ''.join([f'|random_blur(0.5, data_key="query{i}")' for i in range(1, n_queries)]) +
      # '|random_blur(0.1, data_key="query0")|random_solarize(0.2, data_key="query0")' +
      # ''.join([f'|standardize({MEAN_RGB}, {STDDEV_RGB}, data_key="query{i}")' for i in range(n_queries)]) +
      '|keep("reference"' + ''.join(
    [f', "query{i}", "query{i}_box", "query{i}_mask"' for i in range(n_queries)]) + ', "is_l2a")')

  # For MMEARTH
  config.dataset_configs.dataset = 'mm_earth_builder'
  config.dataset_configs.train_split = 'train'

  # Model.
  config.model = ml_collections.ConfigDict()
  config.model.hidden_size = {'Ti': 192,
                              'S': 384,
                              'B': 768,
                              'L': 1024,
                              'H': 1280}[version]
  config.model.patches = ml_collections.ConfigDict()
  config.model.patches.size = [patch, patch]
  config.model.num_heads = {'Ti': 3, 'S': 6, 'B': 12, 'L': 16, 'H': 16}[version]
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
  config.model.head_output_dim = 4096
  config.model.attention_dropout_rate = 0.0
  config.model.dropout_rate = 0.0
  config.model.stochastic_depth = 0.1
  config.model_dtype_str = 'float32'
  config.model.temperature = 0.1
  config.sharpening = 0.05

  # config.initialization = 'imnet_ckt'
  # config.initialization_ckpt = '/home/admin/john/scenic/scenic/projects/loca/imnet_ckpts/loca_vsmall_imnet1k'
  # config.initialization_ckpt = '/home/admin/satellite-loca/scenic/scenic/projects/loca/imnet_ckpts/loca_vsmall_imnet1k'

  # LOCA specific parameters.
  config.n_ref_positions = int((reference_resolution // patch) ** 2)
  config.apply_cluster_loss = True
  config.reference_seqlen = int(0.2 * config.n_ref_positions)  # 20% of 196 is 39
  config.reference_seqlen_selection = 'consecutive'  # or 'unstructured' or 'first'
  config.query_max_seqlen = 70

  # Sen2 Channel-grouping
  config.sen2grouped = True
  config.sen2grouped_maintain_seqlen = True
  # B1:Aerosol = 0, B2:Blue = 1, B3:Green = 2, B4:Red = 3,
  # B5:RedEdge1 = 4, B6:RedEdge2 = 5, B7:RedEdge3 = 6, B8:NIR = 7, B8A:RedEdge4 = 8,
  # B9:WaterVapor = 9, B11:SWIR1 = 10, B12:SWIR2 = 11
  config.sen2changroups = ((1, 2, 3, 7), (4, 5, 6, 8), (10, 11))

  # Multimodal
  config.multimodal = 'early_fuse_s1_to_rgbn'  # None, 'early_fuse_s1_to_rgbn', 'early_fuse_s1_to_all', 'early_concat_s2_and_s1'

  # Training.
  config.max_grad_norm = 1
  config.num_training_epochs = 100
  config.batch_size = 1
  steps_per_epoch = _MMEARTH_TRAIN_SIZE // config.batch_size
  config.rng_seed = 42
  total_steps = config.num_training_epochs * steps_per_epoch

  # Learning rate.
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant * cosine_decay * linear_warmup'
  config.lr_configs.warmup_steps = steps_per_epoch * 15
  config.lr_configs.steps_per_cycle = total_steps
  config.lr_configs.base_learning_rate = 0.001 * config.batch_size / 1024
  config.lr_configs.alpha = 0.01

  # Weight decay.
  config.weight_decay = 0.1

  # Momentum rate scheduler.
  config.momentum_rate = ml_collections.ConfigDict()
  config.momentum_rate.factors = 'constant*cosine_decay'
  config.momentum_rate.steps_per_cycle = total_steps
  config.momentum_rate.base_learning_rate = 0.996
  config.momentum_rate.alpha = 1. / 0.996

  # Logging.
  config.write_summary = True
  config.xprof = True  # Profile using xprof.
  config.checkpoint = True  # Do checkpointing.
  config.checkpoint_steps = 1000
  config.log_summary_steps = 50

  return config

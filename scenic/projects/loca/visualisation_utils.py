import jax.numpy as jnp
from configs.loca_mmearth64_small16 import SENTINEL2_L1C_MEAN, SENTINEL2_L1C_STD, SENTINEL2_L2A_MEAN, SENTINEL2_L2A_STD


def sen2_to_rgb(sen2_img: jnp.ndarray, is_l2a: bool):
  # Reverse normalization
  if is_l2a:
    sen2_img = (sen2_img * jnp.array(SENTINEL2_L2A_STD) + jnp.array(SENTINEL2_L2A_MEAN))
  else:
    sen2_img = (sen2_img * jnp.array(SENTINEL2_L1C_STD) + jnp.array(SENTINEL2_L1C_MEAN))

  # Get RGB channels
  rgb = sen2_img[:, :, [3, 2, 1]]

  # Scale between 0 and 1
  rgb = (rgb - rgb.min(axis=(0, 1))) / (rgb.max(axis=(0, 1)) - rgb.min(axis=(0, 1)))

  # Scale between 0 and 255
  rgb = jnp.array(rgb * 255, dtype="uint8")

  return rgb


def predict_positions(model, train_state, reference: jnp.ndarray, query: jnp.ndarray):
  _, _, r_patch_features, _ = model.flax_model.apply(
    {'params': train_state.ema_params},
    jnp.expand_dims(reference, 0),
    seqlen=-1,  # All patches
    train=False)

  q_loc_pred, _, _, _ = model.flax_model.apply(
    {'params': train_state.params},
    jnp.expand_dims(query, 0),
    inputs_kv=r_patch_features,
    seqlen=-1,  # All patches
    use_pe=False,
    train=False)

  del r_patch_features, _

  return q_loc_pred.squeeze().argmax(axis=1)


def visualize_reference_query_pair(batch, patch_size, batch_idx, query_idx,
                                   ref_idx_kept_tokens=None, ref_idx_kept_groups=None,
                                   q_idx_kept_groups=None):
  """Create visualizations of reference and query images with separate group scaling."""
  import jax.numpy as jnp
  import numpy as np

  # Extract images
  reference = batch['reference'][batch_idx]
  query = batch[f'query{query_idx + 1}'][batch_idx]
  is_l2a = batch.get('is_l2a', True)

  # Channel map and dimensions
  channels_map = {0: [12, 11, 10], 1: [12, 11, 10], 2: [12, 11, 10],
                  3: [2, 2, 2], 4: [2, 2, 2], 5: [0, 0, 0]}
  r_h, r_w, r_c = reference.shape
  q_h, q_w, q_c = query.shape
  r_hp, r_wp = r_h // patch_size, r_w // patch_size
  q_hp, q_wp = q_h // patch_size, q_w // patch_size

  # Group-specific standardization stats (placeholders)
  group_stats = {}
  for g in range(6):
    group_stats[g] = {
      'mean': jnp.array(
        [750.3643697290039, -11.673448681503809, -19.04684937464606, -10.525772696311732, -19.532663009778172,
         -11.754322832475555, -19.187027531976366, -11.729595910348051, -21.294408727496148, 1349.3977794889083,
         1479.9521800379623, 1720.3688077425966, 1899.1848715975957,
         2253.9309600057886, 2808.2001963620132, 3003.424149045887, 3149.5364927329806,
         3110.840562275062, 3213.7636154015954, 2399.086213373806,
         1811.7986415136786]),
      'std': jnp.array(
        [975.1516800116567, 5.1287824026286755, 6.432180454976982, 5.646047662186715, 7.724311760316525,
         5.01231147683192, 6.296322894653895, 5.348371323330351, 7.184819430273331, 2340.2916479338087,
         2375.872101251672, 2256.8997709659416, 2354.181051828758,
         2292.99569489449, 2033.2166835293804, 1920.1736418230105, 1816.6152354201365,
         1988.1938283738782, 1947.9031620588928, 1473.224812450967,
         1390.6781165633136])
    }

  # Setup output images and patch groupings
  query_img = np.zeros((q_h, q_w, 3), dtype=np.float32)
  ref_img = np.zeros((r_h, r_w, 3), dtype=np.float32)

  # Process query groups
  if q_idx_kept_groups is None:
    q_groups = np.zeros((q_hp, q_wp), dtype=int)
  else:
    q_groups = np.array(q_idx_kept_groups[batch_idx]).reshape(q_hp, q_wp)

  # Process reference mask
  ref_mask = np.zeros((r_hp * r_wp), dtype=bool)
  ref_groups = np.zeros((r_hp * r_wp), dtype=int)

  if ref_idx_kept_tokens is not None:
    for i, token in enumerate(ref_idx_kept_tokens):
      if token >= 0 and token < r_hp * r_wp:
        ref_mask[token] = True
        if ref_idx_kept_groups is not None:
          ref_groups[token] = ref_idx_kept_groups[batch_idx][i]
  ref_mask = ref_mask.reshape(r_hp, r_wp)
  ref_groups = ref_groups.reshape(r_hp, r_wp)

  # Unstandardize patch based on group
  def unstandardize(patch, group):
    patch_out = patch.copy()
    for ch in range(9, patch.shape[2]):
      mean, std = group_stats[group]['mean'][ch], group_stats[group]['std'][ch]
      patch_out = patch_out.at[:, :, ch].set(patch[:, :, ch] * std + mean)
    return patch_out

  # Collect values for each group separately
  group_values = {g: [] for g in range(6)}

  # Collect from both query and reference
  for img, groups, hp, wp, h_total, w_total in [
    (query, q_groups, q_hp, q_wp, q_h, q_w),
    (reference, ref_groups, r_hp, r_wp, r_h, r_w)
  ]:
    for i in range(hp):
      for j in range(wp):
        group = int(groups[i, j])
        y, x = i * patch_size, j * patch_size
        h, w = min(patch_size, h_total - y), min(patch_size, w_total - x)
        patch = unstandardize(img[y:y + h, x:x + w], group)

        # For each group, collect values from its specific channels
        for ch in channels_map[group]:
          if ch < img.shape[2]:
            group_values[group].extend(np.array(patch[:, :, ch]).flatten())

  # Calculate min/max for each group separately
  group_min_max = {}
  for g in range(6):
    if group_values[g]:
      # Use percentiles to handle outliers
      group_min_max[g] = {
        'min': np.percentile(group_values[g], 1),
        'max': np.percentile(group_values[g], 99)
      }
      # Ensure we have a reasonable range
      if group_min_max[g]['max'] <= group_min_max[g]['min']:
        group_min_max[g]['min'] = min(group_values[g])
        group_min_max[g]['max'] = max(group_values[g])
    else:
      group_min_max[g] = {'min': 0, 'max': 1}

  # Process query image (no whitening)
  for i in range(q_hp):
    for j in range(q_wp):
      group = int(q_groups[i, j])
      y, x = i * patch_size, j * patch_size
      h, w = min(patch_size, q_h - y), min(patch_size, q_w - x)
      patch = unstandardize(query[y:y + h, x:x + w], group)

      # Get scaling parameters for this group
      g_min = group_min_max[group]['min']
      g_max = group_min_max[group]['max']
      g_scale = g_max - g_min if g_max > g_min else 1

      # Special handling for grayscale channels (groups 3-5)
      if group >= 3:  # Groups 3, 4, 5 are grayscale
        ch = channels_map[group][0]  # Use the first channel (all 3 are the same)
        if ch < q_c:
          ch_data = np.array(patch[:, :, ch])
          # Scale based on this group's min/max
          ch_data = np.clip((ch_data - g_min) / g_scale, 0, 1)
          # Set all RGB channels to the same value for grayscale
          for out_ch in range(3):
            query_img[y:y + h, x:x + w, out_ch] = ch_data
      else:  # RGB channels (groups 0-2)
        for out_ch, in_ch in enumerate(channels_map[group][:3]):
          if in_ch < q_c:
            ch_data = np.array(patch[:, :, in_ch])
            ch_data = np.clip((ch_data - g_min) / g_scale, 0, 1)
            query_img[y:y + h, x:x + w, out_ch] = ch_data

  # Process reference image (with whitening for non-kept patches)
  for i in range(r_hp):
    for j in range(r_wp):
      is_kept = ref_mask[i, j]
      group = int(ref_groups[i, j]) if is_kept else 0
      y, x = i * patch_size, j * patch_size
      h, w = min(patch_size, r_h - y), min(patch_size, r_w - x)
      patch = unstandardize(reference[y:y + h, x:x + w], group)

      # Get scaling parameters for this group
      g_min = group_min_max[group]['min']
      g_max = group_min_max[group]['max']
      g_scale = g_max - g_min if g_max > g_min else 1

      # Special handling for grayscale channels (groups 3-5)
      if group >= 3:  # Groups 3, 4, 5 are grayscale
        ch = channels_map[group][0]  # Use the first channel
        if ch < r_c:
          ch_data = np.array(patch[:, :, ch])
          # Scale based on this group's min/max
          ch_data = np.clip((ch_data - g_min) / g_scale, 0, 1)

          # Whiten non-kept patches
          if not is_kept:
            ch_data = ch_data * 0.3 + 0.7

          # Set all RGB channels to the same value for grayscale
          for out_ch in range(3):
            ref_img[y:y + h, x:x + w, out_ch] = ch_data
      else:  # RGB channels (groups 0-2)
        for out_ch, in_ch in enumerate(channels_map[group][:3]):
          if in_ch < r_c:
            ch_data = np.array(patch[:, :, in_ch])
            ch_data = np.clip((ch_data - g_min) / g_scale, 0, 1)

            # Whiten non-kept patches
            if not is_kept:
              ch_data = ch_data * 0.3 + 0.7

            ref_img[y:y + h, x:x + w, out_ch] = ch_data

  return query_img, ref_img

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

  return q_loc_pred.squeeze().argmax(axis=1)

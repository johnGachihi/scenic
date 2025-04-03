export TFDS_DATA_DIR=/home/admin/john/data/tensorflow_datasets
export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

python -m torch.distributed.launch --nproc_per_node=1 main_finetune.py \
  --model vit_small_simple_cnn_seg \
  --batch_size 8 --accum_iter 16 --blr 0.0001 \
  --epochs 200 --num_workers 3 \
  --input_size 224 --patch_size 16  \
  --weight_decay 0.1 --drop_path 0.0 --reprob 0.0 --mixup 0.0 --cutmix 0.0 --smoothing 0.0 \
  --model_type group_c  \
  --dataset_type sen1floods11 --nb_classes 2 --dropped_bands 0 9 10 \
  --grouped_bands 0 1 2 6 --grouped_bands 3 4 5 7 --grouped_bands 8 9 \
  --train_path /home/fmow-sentinel-filtered-csv/train.csv \
  --test_path /home/fmow-sentinel-filtered-csv/val.csv \
  --output_dir output_dir_small_ft \
  --log_dir log_dir_small_ft \
  --finetune /home/admin/john/scenic/SatMAE/output_dir_small/checkpoint-92.pth
  # --wandb satmae_finetune \
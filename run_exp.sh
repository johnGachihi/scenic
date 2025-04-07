cd SatMAE || exit
python -m torch.distributed.launch --nproc_per_node=1 main_pretrain.py \
  --wandb satmae_pretrain \
  --batch_size 256 --accum_iter 2 --blr 0.0001 \
  --epochs 100 --warmup_epochs 20 --num_workers 16 \
  --input_size 56 --patch_size 4 \
  --mask_ratio 0.75 \
  --model_type group_c \
  --dataset_type mmearth --dataset_version 0.0.6 \
  --dropped_bands 0 9 \
  --grouped_bands 0 1 2 6 --grouped_bands 3 4 5 7 --grouped_bands 8 9 \
  --train_path /home/fmow-sentinel-filtered-csv/train.csv \
  --output_dir output_dir_small \
  --log_dir log_dir_small \


cd ../satmae_pp || exit
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29201 main_pretrain.py \
  --model mae_vit_small \
  --batch_size 128 --accum_iter 4 \
  --epochs 100 --warmup_epochs 20 \
  --input_size 56 --patch_size 4 \
  --mask_ratio 0.75 \
  --model_type group_c \
  --dataset_type mmearth --dataset_version 0.0.6 --dropped_bands 0 9 \
  --grouped_bands 0 1 2 6 --grouped_bands 3 4 5 7 --grouped_bands 8 9 \
  --blr 0.0001 --num_workers 16 \
  --train_path /home/fmow-sentinel/train.csv \
  --output_dir ./output_dir \
  --log_dir ./log_dir

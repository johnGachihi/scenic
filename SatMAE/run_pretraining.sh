python -m torch.distributed.launch --nproc_per_node=1 main_pretrain.py \
  --wandb satmae_pretrain \
  --batch_size 16 --accum_iter 32 --blr 0.0001 \
  --epochs 10 --warmup_epochs 20 --num_workers 8 \
  --input_size 96 --patch_size 8 \
  --mask_ratio 0.75 \
  --model_type group_c \
  --dataset_type mmearth --dropped_bands 0 9 \
  --grouped_bands 0 1 2 6 --grouped_bands 3 4 5 7 --grouped_bands 8 9 \
  --train_path /home/fmow-sentinel-filtered-csv/train.csv \
  --output_dir output_dir \
  --log_dir log_dir \
#  --resume /home/admin/satellite-loca/scenic/SatMAE/output_dir/checkpoint-0.pth
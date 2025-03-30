export CUDA_LAUNCH_BLOCKING=1

python -m torch.distributed.launch --nproc_per_node=1 --master_port=29201 main_pretrain.py \
--model mae_vit_small \
--batch_size 16 --accum_iter 16 \
--epochs 100 --warmup_epochs 20 \
--input_size 96 --patch_size 8 \
--mask_ratio 0.75 \
--model_type group_c \
--dropped_bands 0 9 \
--dataset_type sentinel --dropped_bands 0 9 \
--grouped_bands 0 1 2 6 --grouped_bands 3 4 5 7 --grouped_bands 8 9 \
--blr 0.0001 --num_workers 16 \
--train_path /home/fmow-sentinel/train.csv \
--output_dir ./output_dir \
--log_dir ./output_dir
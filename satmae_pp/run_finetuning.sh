export CUDA_LAUNCH_BLOCKING=1
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29202 main_finetune.py \
--model vit_small_simple_cnn_segmentation \
--batch_size 2 --accum_iter 16 \
--epochs 200 --warmup_epochs 5 \
--input_size 224 --patch_size 16 \
--model_type group_c \
--dataset_type sen1floods11 --dropped_bands 0 9 10 \
--grouped_bands 0 1 2 6 --grouped_bands 3 4 5 7 --grouped_bands 8 9 \
--nb_classes 2 \
--weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.0 --cutmix 0.0 \
--blr 0.0002 \
--num_workers 1 \
--train_path /home/fmow-sentinel/train.csv \
--test_path /home/fmow-sentinel/val.csv \
--output_dir ./output_dir_ft \
--log_dir ./log_dir_ft \
--finetune /home/admin/satellite-loca/scenic/satmae_pp/output_dir/checkpoint-14.pth
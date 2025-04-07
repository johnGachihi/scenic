# export TFDS_DATA_DIR=/home/admin/john/data/tensorflow_datasets
export CUDA_LAUNCH_BLOCKING=1

python -m torch.distributed.launch --nproc_per_node=1 main_finetune.py \
  --model vit_small_simple_cnn_seg \
  --batch_size 64 --accum_iter 2 --blr 0.001 \
  --epochs 200 --num_workers 3 \
  --input_size 112 --patch_size 8  \
  --weight_decay 0.1 --drop_path 0.0 --reprob 0.0 --mixup 0.0 --cutmix 0.0 --smoothing 0.0 \
  --model_type group_c  \
  --dataset_type spacenet1 --nb_classes 2 --grouped_bands 0 1 2 3 4 5 6 7 \
  --train_path /home/fmow-sentinel-filtered-csv/train.csv \
  --test_path /home/fmow-sentinel-filtered-csv/val.csv \
  --output_dir output_dir_small_ft \
  --log_dir log_dir_small_ft \
#  --finetune /home/admin/john/scenic/SatMAE/output_dir_small/checkpoint-92.pth
  # --wandb satmae_finetune \
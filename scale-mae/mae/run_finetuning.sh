export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

python -m torch.distributed.launch --master_port=29501 --nproc_per_node=1 main_finetune.py \
  --model vit_simple_cnn_decoder_small_patch16 \
  --dataset sen1floods11 \
  --batch_size 64 --accum_iter 2 \
  --epochs 200 --warmup_epochs 5 \
  --input_size 224 --patch_size 16 \
  --dropped_bands 0 9 10 \
  --weight_decay 0.1 --drop_path 0.0 --reprob 0.0 --mixup 0.0 --cutmix 0.0 --smoothing 0.0 \
  --output_dir output_dir_ft \
  --log_dir log_dir_ft \
  --num_workers 8 \
  --blr 0.001 \
  --weight_decay 0.1 \
  --finetune /home/admin/satellite-loca/scenic/scale-mae/mae/output_dir/checkpoint-100.pth
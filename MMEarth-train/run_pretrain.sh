export PYTHONPATH=$PYTHONPATH:/home/admin/satellite-loca/scenic/MMEarth-train/MinkowskiEngine
export TFDS_DATA_DIR=/home/admin/john/data/tensorflow_datasets

python  -m torch.distributed.launch --nproc_per_node=1 main_pretrain.py \
        --model convnextv2_tiny \
        --batch_size 512 \
        --update_freq 2 \
        --blr 1.5e-4 \
        --epochs 100 \
        --warmup_epochs 20 \
        --data_dir ../../data/global-lr/data_1M_130_new \
        --output_dir output_dir_tiny \
        --wandb True \
        --wandb_run_name mmearth_tiny_in_s2_out_s1_s2 \
        --wandb_project satellite-loca \
        --loss_aggr uncertainty \
        --auto_resume False \
        --norm_pix_loss True \
        --num_workers 8 \
        --patch_size 8 \
        --input_size 56 \
        --random_crop True \
        --use_orig_stem False \
        --save_ckpt True \
        --no_ffcv True
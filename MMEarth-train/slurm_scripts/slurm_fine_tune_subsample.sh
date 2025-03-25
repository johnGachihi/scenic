#!/bin/bash
#SBATCH --job-name=partitions

#SBATCH --gres=gpu:titanrtx:1
#SBATCH --tasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
# PATH TO SAVE SLURM LOGS
#SBATCH --output=/home/qbk152/vishal/slurm_logs/slurm-training-%A_%a_%x.out
# TOTAL MEMORY PER NODE
#SBATCH --mem=64G 
# NUMBER OF JOBS TO RUN
#SBATCH --array=1-9


#############################################################################################################
# slurm script for finetuning on all classification datasets with one pretraining model, and choosing a 
# subset of the training data
#############################################################################################################



datasets=("m-bigearthnet" "m-so2sat" "m-eurosat")
linear_probe=True
task_type=lp # lp for linear probe, ft for fine-tuning
pretraining=1M-64-full



partitions=("0.01x_train" "0.05x_train" "0.50x_train") # for bigearthnet this corresponds to (200, 1000, 10000) samples
dataset_idx=$(( (SLURM_ARRAY_TASK_ID - 1) / 3))
idx=$(( (SLURM_ARRAY_TASK_ID - 1) % 3 ))
dataset=${datasets[$dataset_idx]}
partition=${partitions[$idx]}
blr=2e-4

output_dir="/projects/dereeco/data/global-lr/ConvNeXt-V2/results_v001/${task_type}-${dataset}_${pretraining}_${partition}"

# if dataset is eurosat set the batch size to 8
if [ $dataset == "m-eurosat" ]; then
    batch_size=8
else
    batch_size=32
fi


python -m  main_finetune \
            --model convnextv2_atto \
            --batch_size $batch_size \
            --update_freq 1 \
            --blr $blr \
            --epochs 100 \
            --warmup_epochs 0 \
            --layer_decay_type 'single' \
            --layer_decay 0.9 \
            --weight_decay 0.3 \
            --drop_path 0.1 \
            --reprob 0.25 \
            --mixup 0. \
            --cutmix 0. \
            --smoothing 0.2 \
            --finetune "/projects/dereeco/data/mmearth_checkpoints_v001/results_1M_64/checkpoint-199.pth" \
            --output_dir "$output_dir" \
            --data_set "$dataset" \
            --linear_probe "$linear_probe" \
            --pretraining "$pretraining"\
            --wandb True \
            --wandb_run_name "subsample--$task_type--$partition--$dataset--$pretraining" \
            --auto_resume False \
            --patch_size 8 \
            --input_size 56 \
            --use_orig_stem False \
            --run_on_test True \
            --save_ckpt True \
            --partition $partition \
            --version 1.0 \
            --num_workers 2 \
            --test_scores_dir /home/qbk152/vishal/MMEarth-train/test_scores/ \
            --geobench_bands_type "full" \
            # --percent $percent




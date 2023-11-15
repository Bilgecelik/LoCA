#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=05:00:00

echo start of OACL run
source activate py3_10
wandb login 71b542c3072e07c51d1184841ffc50858ab2090e
python oacl/pbt_pipeline.py \
--device gpu \
--number_of_experiences 5 \
--number_of_steps 5 \
--perturb_interval 2 \
--number_of_trials 2 \
--quantile_fraction 0.5 \
--resample_probability 0.5 \
--description limited_search_space_lr > results_oacl.txt
echo end of OACL run

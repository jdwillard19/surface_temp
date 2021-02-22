#!/bin/bash -l
#SBATCH --time=23:59:00
#SBATCH --ntasks=8
#SBATCH --mem=20g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=willa099@umn.edu
#SBATCH --output=train_xgb_conus.out
#SBATCH --error=train_xgb_conus.err
#SBATCH --gres=gpu:k40:2
#SBATCH -p k40
source /home/kumarv/willa099/takeme_evaluate.sh

python xgb_error_est.py

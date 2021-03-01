#!/bin/bash -l
#SBATCH --time=23:59:00
#SBATCH --ntasks=8
#SBATCH --mem=20g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=willa099@umn.edu
#SBATCH --output=hypertune_xgb_030121.out
#SBATCH --error=hypertune_xgb_030121.err
#SBATCH --gres=gpu:k40:2
#SBATCH -p k40
source /home/kumarv/willa099/takeme_evaluate.sh

python xgb_hypertune.py

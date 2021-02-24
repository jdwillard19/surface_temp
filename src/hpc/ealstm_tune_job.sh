#!/bin/bash -l
#SBATCH --time=23:59:00
#SBATCH --ntasks=8
#SBATCH --mem=20g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=willa099@umn.edu
#SBATCH --output=EALSTM_tune2.out
#SBATCH --error=EA_LSTM_tune2.err
#SBATCH --gres=gpu:k40:2
#SBATCH -p k40
source /home/kumarv/willa099/takeme_evaluate.sh

python kfold_EALSTM_tune.py

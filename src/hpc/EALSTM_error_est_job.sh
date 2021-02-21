#!/bin/bash -l
#SBATCH --time=23:59:00
#SBATCH --ntasks=8
#SBATCH --mem=20g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=willa099@umn.edu
#SBATCH --output=EALSTM_error_estimation_and_output2.out
#SBATCH --error=EALSTM_error_estimation_and_output2.err
#SBATCH --gres=gpu:k40:2
#SBATCH -p k40
source /home/kumarv/willa099/takeme_evaluate.sh

python EALSTM_error_estimation_and_output2.py

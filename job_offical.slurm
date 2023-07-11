#!/bin/bash
#SBATCH --job-name=clip4clip_model_training
#SBTAHC --partition=debug  
#SBATCH -N 1 
#SBATCH -n 8
#SBTACH --ntasks-per-node=8
#SBATCH --gres=gpu:4
#SBATCH -w node5
#SBATCH --output=../Log/Fusion_Removal/slurm_output/%j.out
#SBATCH --error=../Log/Fusion_Removal/slurm_output/%j.err
#SBATCH -t 8:00:00                            
#SBATCH --mail-type=end
#SBATCH --mail-user=jingxiaolun@sjtu.edu.cn

cd $SLURM_SUBMIT_DIR
sh main_offical.sh





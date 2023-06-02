#!/bin/bash
#SBATCH --job-name=clip4clip_model_training
#SBTAHC --partition=debug  
#SBATCH -N 1 
#SBATCH -n 16
#SBTACH --ntasks-per-node=16
#SBATCH --gres=gpu:8
#SBATCH -w node1
#SBATCH --output=../Log/slurm_output/%j.out
#SBATCH --error=../Log/slurm_output/%j.err
#SBATCH -t 10:00:00                            
#SBATCH --mail-type=end
#SBATCH --mail-user=jingxiaolun@sjtu.edu.cn

cd $SLURM_SUBMIT_DIR
sh main_offical.sh





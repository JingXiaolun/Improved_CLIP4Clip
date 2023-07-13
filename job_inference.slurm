#!/bin/bash
#SBATCH --job-name=se_clip4clip_model_inference
#SBTAHC --partition=debug  
#SBATCH -N 1 
#SBATCH -n 4
#SBTACH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH -w node1
#SBATCH --output=../Log/Frame_Cluster/slurm_output/%j.out
#SBATCH --error=../Log/Frame_Cluster/slurm_output/%j.err
#SBATCH -t 1:00:00                            
#SBATCH --mail-type=end
#SBATCH --mail-user=jingxiaolun@sjtu.edu.cn

cd $SLURM_SUBMIT_DIR
sh main_msrvtt.sh
#sh main_msvd.sh





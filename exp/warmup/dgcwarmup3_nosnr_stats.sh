#!/bin/bash
# 
#SBATCH --job-name=/gscratch/scrubbed/emazuh/adam-compression/dgcwarmup3_nosnr_stats.sh
#SBATCH --account=stf
#SBATCH --partition=ckpt
#
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=12
#SBATCH --time=4:00:00
#SBATCH --gpus-per-node=2
#SBATCH --mem=192G
#SBATCH --requeue
#SBATCH --constraint="[rtx6k|a40|2080ti]"
#
# SBATCH --open-mode=append
#SBATCH --chdir=/gscratch/scrubbed/emazuh/adam-compression
#SBATCH --output=/gscratch/scrubbed/emazuh/experiments/grad/logs/dgcwarmup3_nosnr_stats_%a.log
#SBATCH --error=/gscratch/scrubbed/emazuh/experiments/grad/logs/dgcwarmup3_nosnr_stats_%a.err

#SBATCH --array=0-0

#export PATH=$PATH:/mmfs1/home/emazuh/anaconda3/bin
#export OMP_NUM_THREADS=4


echo $SLURM_ARRAY_JOB_ID_$SLURM_ARRAY_TASK_ID
export PATH=$PATH:/mmfs1/home/emazuh/anaconda3/bin
echo $CONDA_DEFAULT_ENV
module load cuda
module load ompi

HOSTS_FLAG="-H "
for node in $(scontrol show hostnames "$SLURM_JOB_NODELIST"); do
   HOSTS_FLAG="$HOSTS_FLAG$node:$SLURM_NTASKS_PER_NODE,"
done
HOSTS_FLAG=${HOSTS_FLAG%?}
echo $HOSTS_FLAG

mpirun -np $SLURM_NTASKS $HOSTS_FLAG \
    -bind-to none -map-by slot -x NCCL_DEBUG=INFO \
    -x LD_LIBRARY_PATH -x PATH -mca pml ob1 \
    -mca btl ^openib -mca btl_tcp_if_exclude docker0,lo python train.py --configs configs/imagenet/resnet50.py configs/dgc/dgcwm3_nosnr_stats.py configs/dgc/fp16.py configs/dgc/int32.py
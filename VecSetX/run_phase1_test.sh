#!/bin/bash


module load python-data/3.10-24.04
source /projappl/project_2016517/JunjieCheng/junjieenv/bin/activate

export PYTHONPATH=$PYTHONPATH:$(pwd)
export OMP_NUM_THREADS=8


# Create logs directory
mkdir -p logs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run training with torchrun
cd /projappl/project_2016517/JunjieCheng/VecSetX

torchrun --nproc_per_node=1 --master_port=29500 VecSetX/vecset/main_ae.py \
    --batch_size 2 \
    --accum_iter 32 \
    --model learnable_vec1024x16_dim1024_depth24_nb \
    --point_cloud_size 8192 \
    --input_dim 13 \
    --epochs 800 \
    --data_path /scratch/project_2016517/junjie/dataset/repaired_npz \
    --output_dir output/ae/phase1_production \
    --log_dir output/ae/phase1_production \
    --wandb

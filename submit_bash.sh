#!/bin/bash


#SBATCH --time=00:10:00           # Increased time for longer training with larger batches

#SBATCH --mem=256gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=OD-230881
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32        # Increased CPUs for DataLoader workers (H100 can handle more)
#SBATCH --output=slurm-%j.out     # Explicit output file (job ID will be inserted)
#SBATCH --error=slurm-%j.err      # Explicit error file

module load pytorch/2.5.1-py312-cu122-mpi
module load ffmpeg
source $HOME/.venvs/pytorch/bin/activate


# Print job info immediately (helps verify job started)
echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working directory: $(pwd)"
echo "SLURM allocated $SLURM_CPUS_PER_TASK CPUs for this job"
echo "=========================================="


###############################################################NSE TORCHCFD ############################################################
# Training
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

######################################################## TRAINING  ##################################################################

# python3 train_operator_2d.py --config_path configs/pretrain/Customized-Re500-FNO2d-1s-100.yaml --test_ratio 0.25

# python3 train_operator_2d.py --config_path configs/pretrain/Customized-Re500-HFS2d-1s-100.yaml --test_ratio 0.25

# python3 train_operator_2d.py --config_path configs/pretrain/Customized-Re500-WNO2d-1s-100.yaml --test_ratio 0.25

# python3 train_operator_2d.py --config_path configs/pretrain/Customized-Re500-InnerWavelet2D-1s-100.yaml --test_ratio 0.25 --resume_training --resume_ckpt InnerWavelet2D-Re500-1s-100_34001.pt

# python3 train_operator_2d.py --config_path configs/pretrain/Customized-Re500-MultiscaleWavelet2d-1s-100.yaml --test_ratio 0.25

# python3 train_residual_operator_2d.py --config_path configs/pretrain/Customized-Re500-MultiscaleWaveletResidual2d-1s-100.yaml --test_ratio 0.25

# python3 train_operator_2d.py --config_path configs/pretrain/Customized-Re500-MSWTStable2d-1s-100.yaml --test_ratio 0.25

# python3 train_operator_2d.py --config_path configs/pretrain/Customized-Re500-MSWTStableSoft2d-1s-100.yaml --test_ratio 0.25

# python3 train_operator_2d.py --config_path configs/pretrain/Customized-Re500-MSWTStableNormEnergy2d-1s-100.yaml --test_ratio 0.25


# python3 train_operator_2d.py --config_path configs/pretrain/Customized-Re500-InnerWaveletPatching2d-1s-100.yaml --test_ratio 0.25

# python3 train_operator_2d.py --config_path configs/pretrain/Customized-Re500-SAOT2d-1s-100.yaml --test_ratio 0.25 --resume_training --resume_ckpt SAOT2d-Re500-1s-100_63001.pt


######################################################## TESTING  ##################################################################

# python3 eval_operator_2d.py --config_path configs/test/Customized-Re500-FNO2d-05s-test.yaml

# python3 eval_operator_2d.py --config_path configs/test/Customized-Re500-HFS2d-05s-test.yaml

# python3 eval_operator_2d.py --config_path configs/test/Customized-Re500-SAOT2d-05s-test.yaml

# python3 eval_operator_2d.py --config_path configs/test/Customized-Re500-WNO2d-05s-test.yaml

python3 eval_operator_2d.py --config_path configs/test/Customized-Re500-MultiscaleWavelet2d-05s-test.yaml

# python3 eval_operator_2d.py --config_path configs/test/Customized-Re500-InnerWavelet2d-05s-test.yaml

# # python3 eval_operator_2d.py --config_path configs/pretrain/Customized-Re500-WNO2d-1s-100.yaml

# python3 eval_operator_2d.py --config_path configs/test/Customized-Re500-MSWTStable2d-05s-test.yaml

# python3 eval_operator_2d.py --config_path configs/test/Customized-Re500-MSWTStableNormEnergy2d-05s-test.yaml
######################################################## TRAINING 3D ##################################################################

# python3 train_operator.py --config_path configs/pretrain/Customized-Re500-FNO3d-1s-100.yaml --test_ratio 0.25

# python3 train_operator.py --config_path configs/pretrain/Customized-Re500-MultiscaleWavelet3d-1s-100.yaml --test_ratio 0.25


# python3 train_operator_time_dependent.py --config_path configs/pretrain/Customized-Re500-MultiscaleWavelet2d-1s-100-time-conditioned.yaml --test_ratio 0.25
#################################################################################################################################
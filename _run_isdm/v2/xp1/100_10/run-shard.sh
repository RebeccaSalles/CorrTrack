#!/bin/bash

#SBATCH --job-name=v2_100_10_shard
#SBATCH --partition=cpu-dedicated
#SBATCH --qos=dedicated
#SBATCH --time=07-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --threads-per-core=1
#SBATCH --mem=32G
#SBATCH --output=/home/langeb/OAR/out/OAR-v2_100_10_shard_%j.out
#SBATCH --error=/home/langeb/OAR/out/OAR-v2_100_10_shard_%j.out



# display some information about attributed resources
hostname

#cd inria/rebecca/CorrTrack/
#module load conda
source ~/miniforge3/bin/activate rebecca_311
conda activate rebecca_311

# Single-threaded BLAS: avoids oversubscription with the shard threads.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

PYTHONPATH=/storage/simple/projects/iroko-lirmm/CorrTrack python3 -m v2.pipeline /storage/simple/projects/iroko-lirmm/CorrTrack/_run_isdm/v2/xp1/100_10/100_10-pipeline-corrtrack-sharded.json

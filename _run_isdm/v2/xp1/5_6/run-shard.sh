#!/bin/bash

#SBATCH --job-name=v2_5_6_shard
#SBATCH --partition=cpu-dedicated
#SBATCH --qos=dedicated
#SBATCH --time=07-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/langeb/OAR/out/OAR-v2_5_6_shard_%j.out
#SBATCH --error=/home/langeb/OAR/out/OAR-v2_5_6_shard_%j.out



# display some information about attributed resources
hostname

#cd inria/rebecca/CorrTrack/
#module load conda
source ~/miniforge3/bin/activate rebecca_311
conda activate rebecca_311

# Single-threaded BLAS: avoids oversubscription with the 4 shard threads (sharded
# already parallelizes at the series level, multi-threaded BLAS would saturate every core).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

rm -rf /storage/simple/projects/iroko-lirmm/CorrTrack/_run_isdm/v2/xp1/5_6/results

PYTHONPATH=/storage/simple/projects/iroko-lirmm/CorrTrack python3 -m v2.pipeline /storage/simple/projects/iroko-lirmm/CorrTrack/_run_isdm/v2/xp1/5_6/pipeline-run-shard.json

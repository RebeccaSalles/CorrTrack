#!/bin/bash

#SBATCH --job-name=v2_50_10_bf
#SBATCH --partition=cpu-dedicated
#SBATCH --qos=dedicated
#SBATCH --time=07-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --threads-per-core=1
#SBATCH --mem=32G
#SBATCH --output=/home/langeb/OAR/out/OAR-v2_50_10_bf_%j.out
#SBATCH --error=/home/langeb/OAR/out/OAR-v2_50_10_bf_%j.out



# display some information about attributed resources
hostname

#cd inria/rebecca/CorrTrack/
#module load conda
source ~/miniforge3/bin/activate rebecca_311
conda activate rebecca_311

PYTHONPATH=/storage/simple/projects/iroko-lirmm/CorrTrack python3 -m v2.pipeline /storage/simple/projects/iroko-lirmm/CorrTrack/_run_isdm/v2/xp1/50_10/50_10-pipeline-bf.json

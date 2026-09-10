#!/bin/bash

#SBATCH --job-name=00b5_6_32
#SBATCH --partition=cpu-dedicated
#SBATCH --qos=dedicated
#SBATCH --time=07-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --output=/home/langeb/OAR/out/OAR-b5_6_32_%j.out
#SBATCH --error=/home/langeb/OAR/err/OAR-b5_6_32_%j.err


# display some information about attributed resources
hostname

#cd inria/rebecca/CorrTrack/
#module load conda
source ~/miniforge3/bin/activate rebecca_311
conda activate rebecca_311

rm -rf /storage/simple/projects/iroko-lirmm/CorrTrack/_run_isdm/iroko/xp-benoit/b5_6_32/results

CORRTRACK_SKETCH_KERNEL=cython python3  \
        /storage/simple/projects/iroko-lirmm/CorrTrack/run_corrtrack_experiment.py \
	--dataset-config /storage/simple/projects/iroko-lirmm/CorrTrack/_run_isdm/iroko/xp-benoit/b5_6_32/xp1-experiment_dataset_asos.py \
	--param-grid-config /storage/simple/projects/iroko-lirmm/CorrTrack/_run_isdm/iroko/xp-benoit/b5_6_32/xp1-experiment_run_param_grid.py \
	--exec-param-config /storage/simple/projects/iroko-lirmm/CorrTrack/_run_isdm/iroko/xp-benoit/b5_6_32/xp1-experiment_run_exec_param.py \
	--loader datasets.asos_loader:load_dataset \
	--sequential-sketch \
	--sequential-candidates \
	--sequential-validation


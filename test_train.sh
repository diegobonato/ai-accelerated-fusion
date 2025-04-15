#!/bin/bash
#SBATCH --job-name=interactive         # Job name
#SBATCH --output=test%A_%a.out      # Output file for each job (%A: jobID, %a: taskID)
#SBATCH --error=test%A_%a.err        # Error file for each job
#SBATCH --time=2:00:00             # Max run time
#SBATCH --gres=gpu:2                 # Number of GPUs
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40


#SBATCH --account=bsc21
#SBATCH --qos=acc_debug

# Load nvidia module for gpus and the singularity module for the container
module load nvidia-hpc-sdk/24.5  singularity/3.11.5

#singularity exec --nv --bind /gpfs/scratch/bsc21/bsc580556/data_driven:/workspace /home/bsc/bsc580556/nvcr.io_nvidia_modulus_modulus_23.11-2023-11-20-547c0c1003cd.sif python /gpfs/scratch/bsc21/bsc580556/data_driven/afno.py #--id $SLURM_ARRAY_TASK_ID
singularity exec --nv --bind /gpfs/scratch/bsc21/bsc580556/data_driven:/workspace /gpfs/scratch/bsc21/bsc580556/HOME/nvcr.io_nvidia_modulus_modulus_23.11-2023-11-20-547c0c1003cd.sif python /gpfs/scratch/bsc21/bsc580556/data_driven/test.py #--id $SLURM_ARRAY_TASK_ID

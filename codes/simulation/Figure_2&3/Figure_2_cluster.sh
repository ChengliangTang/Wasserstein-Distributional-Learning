#!/bin/sh
#
# Simple "Hello World" submit script for Slurm.
#
# Replace <ACCOUNT> with your account name before submitting.
#
#SBATCH --account=stats      # The account name for the job.
#SBATCH --job-name=CTjob    # The job name.
#SBATCH -c 1                     # The number of cpu cores to use.
#SBATCH --time=0-00:15              # The time the job will take to run.
#SBATCH --mem-per-cpu=16gb        # The memory the job will use per cpu core.
 
module load anaconda
source activate awesomeTensorflow
python -u S01_bootstrap_JASA.py  > temp-$SLURM_ARRAY_TASK_ID.out
 
# End of script

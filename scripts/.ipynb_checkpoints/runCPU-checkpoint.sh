#! /bin/bash -l

#SBATCH -A m3058
#SBATCH -C cpu
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o ../Outputs_Perlmutter/slurm-%j.out # STDOUT
#SBATCH -e ../Outputs_Perlmutter/slurm-%j.err # STDOUT
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=r.henryliu@berkeley.edu

source /global/common/software/desi/desi_environment.sh 23.1 # inherits it
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
module unload desiutil
module load desiutil/3.2.6

echo "compare_combine_EFT_Y_Compton.py"
srun --tres-per-task=cpu:1 python -u compare_combine_EFT_Y_Compton.py

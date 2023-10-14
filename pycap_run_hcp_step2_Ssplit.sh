#!/bin/bash
#SBATCH -J St2_P100
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=12
#SBATCH --partition=pi_anticevic
#SBATCH --time=1-00:00:00
#SBATCH --output="/Studies/CAP_Time_Analytics/cap_codes/slurm_output/St2_P100-%A_%a.out"
#SBATCH --array=901-1000

module load miniconda
source activate /software/env/pycap_env


perm=$SLURM_ARRAY_TASK_ID
Pth=100

basedir=/Studies/

homedir=$basedir/Connectome/subjects/
outdir=$basedir/CAP_Time_Analytics/results/Ssplit$perm/
mkdir $outdir
datadir=$basedir/CAP_Time_Analytics/groupdata/Ssplit$perm/
mkdir $datadir
pscalarfilen=$basedir/Connectome/Parcellations/CABNP/CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR_ReorderedByNetworks_Zeroes.pscalar.nii
sublistfilen=$basedir/CAP_Time_Analytics/sublist/hcp_n337.csv

python $basedir/CAP_Time_Analytics/cap_codes/pyCAP/pyCAP/pycap_Ssplit_run_hcp_step2_cluster2cap.py \
--homedir $homedir \
--outdir $outdir \
--datadir $datadir \
--sublistfilen $sublistfilen \
--pscalarfilen $pscalarfilen \
--mink 2 \
--maxk 15 \
--maxiter 1000 \
--kmethod silhouette \
--savecapimg n \
--gsr y \
--unit p \
--seedtype seedfree \
--randTthreshold $Pth \
--scrubbing y \
--runorder 2134 \
--motiontype fd \
--motionthreshold 0.5 \
--motiondisplay n \
--ndummy 100 




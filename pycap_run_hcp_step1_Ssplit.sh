#!/bin/bash
#SBATCH -J St1_k2_P100
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=12
#SBATCH --partition=day
#SBATCH --time=1-00:00:00
#SBATCH --output="/Studies/CAP_Time_Analytics/cap_codes/slurm_output/St1_k2_P100-%A_%a.out"
#SBATCH --array=901-1000

module load miniconda
source activate /software/env/pycap_env


basedir=/Studies/

for perm in $SLURM_ARRAY_TASK_ID
do
    for Pth in 100
    do
        homedir=$basedir/Connectome/subjects/
        outdir=$basedir/CAP_Time_Analytics/results/Ssplit$perm/
        mkdir $outdir
        datadir=$basedir/CAP_Time_Analytics/groupdata/Ssplit$perm/
        mkdir $datadir
        pscalarfilen=$basedir/Connectome/Parcellations/CABNP/CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR_ReorderedByNetworks_Zeroes.pscalar.nii
        sublistfilen=$basedir/CAP_Time_Analytics/sublist/hcp_n337.csv

        for i in 2 #3 4 5 6 7 8 9 10 11 12 13 14 15
        do
            python $basedir/CAP_Time_Analytics/cap_codes/pyCAP/pyCAP/pycap_Ssplit_run_hcp_step1_findk.py \
            --homedir $homedir \
            --outdir $outdir \
            --datadir $datadir \
            --sublistfilen $sublistfilen \
            --pscalarfilen $pscalarfilen \
            --ncluster $i \
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
            --ndummy 100 \
            --subsplittype random
        done
    done
done


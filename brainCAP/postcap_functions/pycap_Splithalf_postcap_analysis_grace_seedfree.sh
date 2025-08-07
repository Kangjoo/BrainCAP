#!/bin/bash

#SBATCH -J postcap_seedfree
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=8
#SBATCH --partition=day
#SBATCH --time=1-00:00:00
#SBATCH --output="/Studies/CAP_Time_Analytics/cap_codes/slurm_output/postcap_seedfree-%A.out"


module load miniconda
source activate /software/env/pycap_env



####################################################
#
#                Post-CAP analysis
#
####################################################

basedir=/Studies/


# Initial information required before running any CAP codes
homedir=$basedir/Connectome/subjects/
outdir=$basedir/CAP_Time_Analytics/results/Splithalf_postcap/
mkdir $outdir
datadirtag="$basedir/CAP_Time_Analytics/results/"
pscalarfilen=/$basedir/Connectome/Parcellations/CABNP/CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR_ReorderedByNetworks_Zeroes.pscalar.nii
sublistfilen=$basedir/CAP_Time_Analytics/sublist/hcp_n337.csv
parcelinfofilen=$basedir/CAP_Time_Analytics/parcellation/CABNP_info.csv
maxperm = 10



# Information needed to run Step 12
# These information can be obtained after running <Step 8: Identify subsets of population using time-analytics>
ksubgroupfilen=$basedir/CAP_Time_Analytics/results/Splithalf_postcap/gsr_seedfree/P100.0_k_solution_subcounts_allcTh.csv
subgroupfilen=$basedir/CAP_Time_Analytics/results/Splithalf_postcap/gsr_seedfree/P100.0_FO_mDT_vDT_subgroup_30nf.csv

# Information needed to run Step 12 
# These information can be obtained after running NBM analysis on behavioral data
neuralpcafilen=$basedir/CAP_Time_Analytics/results/Splithalf_postcap/gsr_seedfree/P100.0_FO_mDT_vDT_neuralPCs_30nf.csv
behavtag="All_All_flip_dropAS_30nf"
behavpcn=27
behavfilen=/CAP_behavior_prep_RESTRICTED_hcp_n337_All_All_flip_dropAS_30nf.tsv
behavpcafilen=$basedir/CAP_Time_Analytics/results/Connectome_All_All_flip_dropAS_30nf_n337/analysis/results/NBRIDGE_Connectome_All_All_flip_dropAS_30nf_n337_BehaviorPCAScores.tsv





# ####################################################
# #   Step 1: collect data
# ####################################################

for st in 100 #st=100 indicates using 100% of datapoints (seedfree)
do
    python $basedir/CAP_Time_Analytics/cap_codes/pyCAP/pyCAP/pycap_Splithalf_postcap_step1_colldata.py \
    --homedir $homedir \
    --outdir $outdir \
    --datadirtag $datadirtag \
    --sublistfilen $sublistfilen \
    --pscalarfilen $pscalarfilen \
    --gsr y \
    --unit p \
    --minperm 1 \
    --maxperm $maxperm \
    --seedtype seedfree \
    --randTthreshold $st
done


####################################################
#   Step 2: Plot QC and group level stats
####################################################

python $basedir/CAP_Time_Analytics/cap_codes/pyCAP/pyCAP/pycap_Splithalf_postcap_step2_plotqcgroup.py \
--homedir $homedir \
--outdir $outdir \
--sublistfilen $sublistfilen \
--pscalarfilen $pscalarfilen \
--gsr y \
--unit p \
--minperm 1 \
--maxperm $maxperm \
--seedtype seedfree \
--randTthresholdrange 100 \
--ncapofinterest 4 5



####################################################
#   Step 3: determine a set of basis CAPs
####################################################

python $basedir/CAP_Time_Analytics/cap_codes/pyCAP/pyCAP/pycap_Splithalf_postcap_step3_basis_caps.py \
--homedir $homedir \
--outdir $outdir \
--datadirtag $datadirtag \
--sublistfilen $sublistfilen \
--pscalarfilen $pscalarfilen \
--gsr y \
--unit p \
--minperm 1 \
--maxperm $maxperm \
--seedtype seedfree \
--standardTthreshold 100 \
--basismethod hac \
--basis_k_range 4 5



#####################################################################
#   Step 4: spatial reliability of basis CAPs (marginal distribution analysis)
#####################################################################

for stdk in 4 5
do
    python $basedir/CAP_Time_Analytics/cap_codes/pyCAP/pyCAP/pycap_Splithalf_postcap_step4_basis_45caps_margdist.py \
    --homedir $homedir \
    --outdir $outdir \
    --datadirtag $datadirtag \
    --sublistfilen $sublistfilen \
    --pscalarfilen $pscalarfilen \
    --gsr y \
    --unit p \
    --minperm 1 \
    --maxperm $maxperm \
    --seedtype seedfree \
    --standardTthreshold 100 \
    --randTthresholdrange 100 \
    --basismethod hac \
    --basis_k_range $stdk
done



#####################################################################
#   Step 5: Basis CAPs vs CAB-NP
#####################################################################

for stdk in 4 5
do
    python $basedir/CAP_Time_Analytics/cap_codes/pyCAP/pyCAP/pycap_Splithalf_postcap_step5_basis_45caps_vs_cabnp.py \
    --homedir $homedir \
    --outdir $outdir \
    --datadirtag $datadirtag \
    --sublistfilen $sublistfilen \
    --pscalarfilen $pscalarfilen \
    --parcelinfofilen $parcelinfofilen \
    --gsr y \
    --unit p \
    --minperm 1 \
    --maxperm $maxperm \
    --seedtype seedfree \
    --standardTthreshold 100 \
    --randTthresholdrange 100 \
    --basismethod hac \
    --basis_k_range $stdk
done



####################################################
#   Step 6: Identify subsets of population contribution to k=4 or 5 solutions
####################################################

python $basedir/CAP_Time_Analytics/cap_codes/pyCAP/pyCAP/pycap_Splithalf_postcap_step6_subset_ksolution.py \
--homedir $homedir \
--outdir $outdir \
--datadirtag $datadirtag \
--sublistfilen $sublistfilen \
--pscalarfilen $pscalarfilen \
--gsr y \
--unit p \
--minperm 1 \
--maxperm $maxperm \
--seedtype seedfree \
--randTthresholdrange 100 \
--classTh 1 2 \
--basis_k_range 4 5



####################################################
#   Step 7: Test-retest (Day 1 vs Day 2) Time analytics Data (e.g.use 5-basis CAPs) 
####################################################

python $basedir/CAP_Time_Analytics/cap_codes/pyCAP/pyCAP/pycap_Splithalf_postcap_step7_TRT_timeanalytics_data.py \
--homedir $homedir \
--outdir $outdir \
--datadirtag $datadirtag \
--sublistfilen $sublistfilen \
--pscalarfilen $pscalarfilen \
--gsr y \
--unit p \
--minperm 1 \
--maxperm $maxperm \
--seedtype seedfree \
--randTthresholdrange 100 \
--basis_k_range 5




####################################################
#   Step 8: Identify subsets of population using time-analytics 
####################################################

python $basedir/CAP_Time_Analytics/cap_codes/pyCAP/pyCAP/pycap_Splithalf_postcap_step8_subset_timeanalytics.py \
--homedir $homedir \
--outdir $outdir \
--datadirtag $datadirtag \
--sublistfilen $sublistfilen \
--pscalarfilen $pscalarfilen \
--gsr y \
--unit p \
--minperm 1 \
--maxperm $maxperm \
--seedtype seedfree \
--randTthresholdrange 100 \
--basis_k_range 5



####################################################
#   Step 9: Test-retest (Day 1 vs Day 2) Time analytics Plot 
####################################################

python $basedir/CAP_Time_Analytics/cap_codes/pyCAP/pyCAP/pycap_Splithalf_postcap_step9_TRT_timeanalytics_plot.py \
--homedir $homedir \
--outdir $outdir \
--datadirtag $datadirtag \
--sublistfilen $sublistfilen \
--pscalarfilen $pscalarfilen \
--subgroupfilen $subgroupfilen \
--gsr y \
--unit p \
--minperm 1 \
--maxperm $maxperm \
--seedtype seedfree \
--randTthresholdrange 100 \
--classTh 30 \
--basis_k_range 5



####################################################
#   Step 10: Individual reliability Time analytics (heatmap rank-ordered subjects) 
#   Submit to bigmem, for each neural measure seperately. 
####################################################

python $basedir/CAP_Time_Analytics/cap_codes/pyCAP/pyCAP/pycap_Splithalf_postcap_step10_indpermrel_timeanalytics.py \
--homedir $homedir \
--outdir $outdir \
--datadirtag $datadirtag \
--sublistfilen $sublistfilen \
--pscalarfilen $pscalarfilen \
--subgroupfilen $subgroupfilen \
--gsr y \
--unit p \
--minperm 1 \
--maxperm $maxperm \
--seedtype seedfree \
--randTthresholdrange 100 \
--classTh 30 \
--basis_k_range 5



####################################################
#   Step 11: Positive-Negative state comparisons of Time analytics 
####################################################

python $basedir/CAP_Time_Analytics/cap_codes/pyCAP/pyCAP/pycap_Splithalf_postcap_step11_posneg_timeanalytics.py \
--homedir $homedir \
--outdir $outdir \
--datadirtag $datadirtag \
--sublistfilen $sublistfilen \
--pscalarfilen $pscalarfilen \
--gsr y \
--unit p \
--minperm 1 \
--maxperm $maxperm \
--seedtype seedfree \
--randTthresholdrange 100 \
--basis_k_range 5



####################################################
#   Step 12: Subgroups of k=4 or 5 solutions vs bahavior 
####################################################

python $basedir/CAP_Time_Analytics/cap_codes/pyCAP/pyCAP/pycap_Splithalf_postcap_step12_subset_ksolution_to_behavior.py \
--homedir $homedir \
--outdir $outdir \
--datadirtag $datadirtag \
--sublistfilen $sublistfilen \
--pscalarfilen $pscalarfilen \
--neuralpcafilen $neuralpcafilen \
--ksubgroupfilen $ksubgroupfilen \
--neuralsubgroupfilen $subgroupfilen \
--behavtag $behavtag \
--behavfilen $behavfilen \
--behavpcafilen $behavpcafilen \
--neuralpcafilen $neuralpcafilen \
--gsr y \
--unit p \
--minperm 1 \
--maxperm $maxperm \
--seedtype seedfree \
--randTthresholdrange 100 \
--classTh 30 \
--behavpcn $behavpcn \
--basis_k_range 5

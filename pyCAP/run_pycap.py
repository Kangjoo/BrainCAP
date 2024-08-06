import subprocess
import os
#from time import time, ctime
import logging
from datetime import datetime
from pathlib import Path
import argparse

def remove_directory_tree(start_directory: Path):
    """Recursively and permanently removes the specified directory, all of its
    subdirectories, and every file contained in any of those folders."""
    for path in start_directory.iterdir():
        if path.is_file():
            path.unlink()
        else:
            remove_directory_tree(path)
    start_directory.rmdir()


#source activate /gpfs/gibbs/pi/n3/software/env/pycap_env


def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")

def file_path(path):
    if os.path.exists(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"file {path} does not exist!")

def local_path(path):
    if path[0] != '/':
        return path
    else:
        raise argparse.ArgumentTypeError(f"{path} must be a local path from the specified sessions_dir!")


parser = argparse.ArgumentParser()

parser.add_argument("--sessionsfolder", type=dir_path, required=True, help="Path to directory containing session data.")
parser.add_argument("--outputfolder", type=str, required=True , help="Output directory for analysis results.")
parser.add_argument("--ptemplate", type=file_path, required=False , help="Path to parcellation template if using parcellated data.")
parser.add_argument("--sessionslist", type=file_path, required=True, help="Path to text file containing line-seperated list of sessions.")
parser.add_argument("--inputbold", type=str, required=True, help="Local path inside each session's folder to the input BOLD data. If there are multiple BOLDs per session, " \
                    "then this will be the path of the saved concatenated BOLD file. Multiple BOLDs must be specified with the '--inputbolds' flag. ")
parser.add_argument("--inputbolds", type=str, required=False, help="Comma seperated list of BOLD files to be concatenated if data has multiple BOLDs.")
parser.add_argument("--inputmotion", type=str, required=False, help="Local path inside each session's folder to the input motion data if using scrubbing. If there are multiple BOLDs per session, " \
                    "then this will be the path of the saved concatenated motion file. Multiple motion files must be specified with the '--inputmotions' flag. ")
parser.add_argument("--inputmotions", type=str, required=False, help="Comma seperated list of BOLD files to be concatenated if data has multiple BOLDs.")
parser.add_argument("--ndummy", type=int, default=0, help="Number of initial dummy frames to remove")
parser.add_argument("--step", type=str, required=True, help="Comma seperated list of PyCap steps to run")
parser.add_argument("--overwrite", type=str, default="no", help="Whether to overwrite existing data")
parser.add_argument("--scheduler", type=str, default="none", help="What scheduler to use (SLURM, PBS, or none). Requires relevant scheduler to be installed.")
parser.add_argument("--kvals", default="2-15", help="Either a range (eg. 2-5) or comma-seperated list (2,3,4,5) of k-values to use") #NEED TO ADD PARSING FOR THIS
args = parser.parse_args()

#input parameters
sessionsfolder="/gpfs/gibbs/pi/n3/Studies/Connectome/subjects/"
analysisfolder="/gpfs/gibbs/pi/n3/Studies/CAP_Time_Analytics/devtest/prep_test/"
parcfile="/gpfs/gibbs/pi/n3/Studies/Connectome/Parcellations/CABNP/CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR_ReorderedByNetworks_Zeroes.pscalar.nii"
sessions="/gpfs/gibbs/pi/n3/Studies/CAP_Time_Analytics/sublist/hcp_n25.csv"

bolds="images/functional/bold2_Atlas_MSMAll_hp2000_clean_res-WB_demean.dtseries.nii," \
    "images/functional/bold1_Atlas_MSMAll_hp2000_clean_res-WB_demean.dtseries.nii," \
    "images/functional/bold4_Atlas_MSMAll_hp2000_clean_res-WB_demean.dtseries.nii," \
    "images/functional/bold3_Atlas_MSMAll_hp2000_clean_res-WB_demean.dtseries.nii"

motionfiles="images/functional/movement/bold2.bstats," \
    "images/functional/movement/bold1.bstats," \
    "images/functional/movement/bold4.bstats," \
    "images/functional/movement/bold3.bstats"

datafile="images/functional/bold2143_test.dtseries.nii"
motionfile="images/functional/movement/bold2143_test.bstats"
overwrite="yes"
ndummy=100
steps="pycap"

timestamp = datetime.now().strftime("%Y-%m-%d_%H.%M.%S.%f")

# if overwrite=='yes' and os.path.exists(analysisfolder):
#     remove_directory_tree(Path(analysisfolder))


if not os.path.exists(analysisfolder):
    os.makedirs(analysisfolder)

maxiter = 1000

#scheduler parameters
job_name = '25test'
mem = '8G'
cpus = 4
partition = 'pi_anticevic'
time = '1:00:00'

nsplits = 2
k_range = [2, 5]
par_k = 1 #Not currently used

#calculated parameters
array = [1, nsplits]
k_vals = list(range(k_range[0],k_range[1]+1))
k_n = len(k_vals)
results_dir = os.path.join(analysisfolder, 'results/')
group_dir = os.path.join(analysisfolder, 'groupdata/')
outdir = os.path.join(results_dir, 'Ssplit_')
datadir = os.path.join(group_dir, 'Ssplit_')


step_list = steps.split(',')

# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s - %(message)s',
#                     filename=f"{analysisfolder}run_pycap.log",
#                     filemode='w')
# console = logging.StreamHandler()
# formatter = logging.Formatter('%(asctime)s - %(message)s')
# console.setFormatter(formatter)
# logging.getLogger('').addHandler(console)

#PyCap setup
if 'pre_pycap' in step_list:
    slurm_out = f"/gpfs/gibbs/pi/n3/Studies/CAP_Time_Analytics/time-analytics/slurm_output/prep_{job_name}-{timestamp}.out"
    for path in [analysisfolder, results_dir, group_dir]:
        if not os.path.exists(path):
            os.makedirs(path)

    command = "#!/bin/bash\n"
    command += f"#SBATCH -J setup_{job_name}\n"
    command += f"#SBATCH --mem-per-cpu=40G\n" #If dense, needs more data than if parc
    command += f"#SBATCH --cpus-per-task=1\n"
    command += f"#SBATCH --partition={partition}\n"
    command += f"#SBATCH --time=2:00:00\n"
    command += f"#SBATCH --output={slurm_out}\n"
    command += f"#SBATCH --nodes=1\n"
    command += f"#SBATCH --ntasks=1\n"

    #Attempt to concatenate bolds if multiple specified
    if bolds is not None:
        command += f"python /gpfs/gibbs/pi/n3/Studies/CAP_Time_Analytics/time-analytics/pyCAP/pyCAP/pycap_concatenate.py "
        command += f"--sessions_list {sessions} "
        command += f"--sessions_dir {sessionsfolder} "
        command += f"--bolds {bolds} "
        command += f"--output {datafile} "
        command += f"--ndummy {ndummy} "
        command += f"--overwrite {overwrite} "
        command += f"--logpath {analysisfolder}run_pycap.log "

        if motionfiles is not None:
            command += f"--motionfiles {motionfiles} "
            command += f"--outputmotion {motionfile} "

        command += f"\n"

        command += f"wait\n"

    for split in range(nsplits):
        if not os.path.exists(f'{outdir}{split+1}'):
            os.makedirs(f'{outdir}{split+1}')
        if not os.path.exists(f'{datadir}{split+1}'):
            os.makedirs(f'{datadir}{split+1}')

        command += f"python /gpfs/gibbs/pi/n3/Studies/CAP_Time_Analytics/time-analytics/pyCAP/pyCAP/pycap_Ssplit_run_hcp.py "
        command += f"--homedir {sessionsfolder} "
        command += f"--outdir {outdir}{split+1}/ "
        command += f"--datadir {datadir}{split+1} "
        command += f"--inputdata {datafile} "
        command += f"--sublistfilen {sessions} "
        command += f"--pscalarfilen {parcfile} "
        command += f"--ncluster 0 "
        command += f"--mink {k_range[0]} "
        command += f"--maxk {k_range[1]} "
        command += f"--maxiter {maxiter} "
        command += f"--kmethod silhouette "
        command += f"--savecapimg n "
        command += f"--gsr y "
        command += f"--unit p "
        command += f"--seedtype seedfree "
        command += f"--randTthreshold 100 "
        command += f"--scrubbing y "
        #command += f"--runorder 2134 "
        command += f"--motionfile {motionfile} "
        command += f"--motiontype fd "
        command += f"--motionthreshold 0.5 "
        command += f"--motiondisplay n "
        command += f"--ndummy 100 "
        command += f"--step prep "
        command += f"--subsplittype random \n"

    serr = subprocess.STDOUT
    sout = subprocess.PIPE

    run = subprocess.Popen(
            "sbatch", shell=True, stdin=subprocess.PIPE, stdout=sout, stderr=serr, close_fds=True
        )

    run.stdin.write((command).encode("utf-8"))
    out = run.communicate()[0].decode("utf-8")
    run.stdin.close()

    job_id_prep = out.split("Submitted batch job ")[1]
    print("Running PyCAP with prep")
    print("Follow prep command progress in:")
    print(f"{slurm_out}")
    run_prep = True
else:
    print("Running PyCAP without prep")
    job_id_prep = None
    run_prep = False

print('\n')

#PyCap
if 'pycap' in step_list:
    slurm_out = f"/gpfs/gibbs/pi/n3/Studies/CAP_Time_Analytics/time-analytics/slurm_output/main_{job_name}-{timestamp}.out"

    command = "#!/bin/bash\n"
    command += f"#SBATCH -J {job_name}\n"
    command += f"#SBATCH --mem-per-cpu={mem}\n"
    command += f"#SBATCH --cpus-per-task={cpus}\n"
    command += f"#SBATCH --partition={partition}\n"
    command += f"#SBATCH --time={time}\n"
    command += f"#SBATCH --output={slurm_out}\n"
    command += f"#SBATCH --array={array[0]}-{array[1]}\n"
    command += f"#SBATCH --nodes=1\n"
    command += f"#SBATCH --ntasks={k_n}\n"


    command += "PERM=${SLURM_ARRAY_TASK_ID} \n"

    #Pycap step 1
    for k in k_vals:
        command += f"srun --ntasks=1 python /gpfs/gibbs/pi/n3/Studies/CAP_Time_Analytics/time-analytics/pyCAP/pyCAP/pycap_Ssplit_run_hcp.py "
        command += f"--homedir {sessionsfolder} "
        command += f"--outdir {outdir}$PERM/ "
        command += f"--datadir {datadir}$PERM/ "
        command += f"--inputdata {datafile} "
        command += f"--sublistfilen {sessions} "
        command += f"--pscalarfilen {parcfile} "
        command += f"--ncluster {k} "
        command += f"--mink {k_range[0]} "
        command += f"--maxk {k_range[1]} "
        command += f"--maxiter {maxiter} "
        command += f"--kmethod silhouette "
        command += f"--savecapimg n "
        command += f"--gsr y "
        command += f"--unit p "
        command += f"--seedtype seedfree "
        command += f"--randTthreshold 100 "
        command += f"--scrubbing y "
        #command += f"--runorder 2134 "
        command += f"--motionfile {motionfile} "
        command += f"--motiontype fd "
        command += f"--motionthreshold 0.5 "
        command += f"--motiondisplay n "
        command += f"--ndummy 100 "
        command += f"--step step1 "
        command += f"--subsplittype random &\n"

    command += "wait\n"

    #Pycap step 2
    command += f"python /gpfs/gibbs/pi/n3/Studies/CAP_Time_Analytics/time-analytics/pyCAP/pyCAP/pycap_Ssplit_run_hcp.py "
    command += f"--homedir {sessionsfolder} "
    command += f"--outdir {outdir}$PERM/ "
    command += f"--datadir {datadir}$PERM "
    command += f"--inputdata {datafile} "
    command += f"--sublistfilen {sessions} "
    command += f"--pscalarfilen {parcfile} "
    command += f"--ncluster {k} "
    command += f"--mink {k_range[0]} "
    command += f"--maxk {k_range[1]} "
    command += f"--maxiter {maxiter} "
    command += f"--kmethod silhouette "
    command += f"--savecapimg n "
    command += f"--gsr y "
    command += f"--unit p "
    command += f"--seedtype seedfree "
    command += f"--randTthreshold 100 "
    command += f"--scrubbing y "
    # command += f"--runorder 2134 "
    command += f"--motionfile {motionfile} "
    command += f"--motiontype fd "
    command += f"--motionthreshold 0.5 "
    command += f"--motiondisplay n "
    command += f"--ndummy 100 "
    command += f"--step step2 "
    command += f"--subsplittype random \n"

    serr = subprocess.STDOUT
    sout = subprocess.PIPE

    
    

    if run_prep:
        sbatch = f"sbatch --dependency afterok:{job_id_prep}"
        print("Main PyCap job will wait on `prep` to finish")
        print("Once complete, follow main command progress in:")
    else:
        print("Follow main command progress in:")
        sbatch = f"sbatch"

    print(f"{slurm_out}")
    

    run = subprocess.Popen(
            sbatch, shell=True, stdin=subprocess.PIPE, stdout=sout, stderr=serr, close_fds=True
        )

    run.stdin.write((command).encode("utf-8"))
    run.stdin.close()
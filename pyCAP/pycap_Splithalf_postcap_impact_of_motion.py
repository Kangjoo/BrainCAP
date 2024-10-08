#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# Created By  : Kangjoo Lee (kangjoo.lee@yale.edu)
# Last Updated: 08/06/2024
# -------------------------------------------------------------------------

# =========================================================================
#                    --   Run pipeline template  --
#      Analysis of Co-Activation Patterns(CAP) in fMRI data (HCP-YA)
# =========================================================================



# Imports
import shutil
import math
import h5py
import os
from os.path import exists as file_exists
import random
import sklearn.model_selection
import numpy as np
import argparse
import itertools
import pandas as pd
import logging
from pycap.pyCAP.pycap_functions.pycap_loaddata import *
from pycap_functions.pycap_frameselection import *
from pycap_functions.pycap_gen import *
from pycap_functions.pycap_datasplit import *
import seaborn as sns
import matplotlib.pyplot as plt





def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--scrubbing", type=str, help="Use scrugging or not (y/n)")
parser.add_argument("-d", "--ndummy", type=int, help="Number of dummy scans to remove")
parser.add_argument("-g", "--gsr", type=str, help="(y/n)")
parser.add_argument("-hd", "--homedir", type=dir_path, help="Home directory path")
parser.add_argument("-od", "--outdir", type=dir_path, help="Output directory path")
parser.add_argument("-m", "--motiontype", type=str, help="(dvarsm,dvarsme,fd)")
parser.add_argument("-r", "--runorder", type=str, help="Order or runs to be concatenated")
parser.add_argument("-s", "--seedtype", type=str, help="(seedfree/seedbased)")
parser.add_argument("-si", "--seedname", type=str, help="Seed name")
parser.add_argument("-sp", "--seedthreshtype", type=str, help="(T/P)")
parser.add_argument("-st", "--seedthreshold", type=float, help="Signal threshold")
parser.add_argument("-sl", "--sublistfilen", dest="sublistfilen", required=True,
                    help="Subject list filename", type=lambda f: open(f))
parser.add_argument("-ts", "--subsplittype", type=str, help="random/days")
parser.add_argument("-p", "--pscalarfilen", dest="pscalarfile", required=True,
                    help="Pscalar filename", type=lambda f: open(f))
parser.add_argument("-u", "--unit", type=str, help="(p/d)")  # parcel or dense
parser.add_argument("-v", "--motionthreshold", type=float, help="Motion threshold")
parser.add_argument("-w", "--motiondisplay", type=str,
                    help="Display motio parameter or not (y/n)")
parser.add_argument("-tt", "--randTthreshold", type=float, help="Random Time Signal threshold")                    
parser.add_argument("-dd", "--datadir", type=dir_path, help="Concatenated Data directory path")


                    
args = parser.parse_args()  # Read arguments from command line




# -------------------------------------------
#           Setup input parameters
# -------------------------------------------


class Param:
    pass


param = Param()

# - parameters for data selection
if args.gsr == "y":
    param.gsr = "gsr"
elif args.gsr == "n":
    param.gsr = "nogsr"
param.unit = args.unit
if param.unit == "d":
    param.sdim = 91282
elif param.unit == "p":
    param.sdim = 718
param.tdim = 4400
param.subsplittype = args.subsplittype

# - parameters for seed signal selection
param.seedtype = args.seedtype
if param.seedtype == "seedbased":
    param.seedIDname = args.seedname
    param.seedID = eval(param.seedIDname)
    param.eventcombine = args.eventcombine
    param.eventtype = args.eventtype
    param.sig_thresholdtype = args.seedthreshtype
    param.sig_threshold = args.seedthreshold
elif param.seedtype == "seedfree":
    param.seedIDname = param.seedtype
    param.randTthreshold = args.randTthreshold


# - parameters for motion scrubbing
param.scrubbing = args.scrubbing
param.motion_type = args.motiontype
param.motion_threshold = args.motionthreshold  # dvarsmt=[3.0], dvarsmet=[1.6], fdr=[0.5]
param.motion_display = args.motiondisplay
param.n_dummy = args.ndummy
param.run_order = list(args.runorder)




# -------------------------------------------
#              Setup input data
# -------------------------------------------


class FileIn:
    pass


filein = FileIn()
filein.homedir = args.homedir
sl = pd.read_csv(args.sublistfilen.name, header=0).values.tolist()
filein.sublist = list(itertools.chain.from_iterable(sl))
filein.pscalar_filen = args.pscalarfile.name


if param.unit == "d":
    if param.gsr == "nogsr":
        filein.fname = "bold2143_Atlas_MSMAll_hp2000_clean_demean-100f.dtseries.nii"
    elif param.gsr == "gsr":
        filein.fname = "bold2143_Atlas_MSMAll_hp2000_clean_res-WB_demean-100f.dtseries.nii"
elif param.unit == "p":
    if param.gsr == "nogsr":
        filein.fname = "bold2143_Atlas_MSMAll_hp2000_clean_demean-100f_CAB-NP_Parcels_ReorderedByNetwork.ptseries.nii"
    elif param.gsr == "gsr":
        filein.fname = "bold2143_Atlas_MSMAll_hp2000_clean_res-WB_demean-100f_CAB-NP_Parcels_ReorderedByNetwork.ptseries.nii"

# -------------------------------------------
#            Setup output directory
# -------------------------------------------

if param.seedtype == "seedbased":
    filein.outpath = args.outdir + param.gsr + "_" + param.seedIDname + \
        "/" + param.sig_thresholdtype + str(param.sig_threshold) + "/"
    filein.datadir = args.datadir + param.gsr + "_" + param.seedIDname + \
        "/" + param.sig_thresholdtype + str(param.sig_threshold) + "/"
elif param.seedtype == "seedfree":
    filein.outpath = args.outdir + param.gsr + "_" + \
        param.seedIDname + "/P" + str(param.randTthreshold) + "/"
    filein.datadir = args.datadir + param.gsr + "_" + \
        param.seedIDname + "/P" + str(param.randTthreshold) + "/"
filein.outdir = filein.outpath

isExist = os.path.exists(filein.outdir)
if not isExist:
    os.makedirs(filein.outdir)
del isExist

isExist = os.path.exists(filein.datadir)
if not isExist:
    os.makedirs(filein.datadir)
del isExist


# -------------------------------------------
#            Setup log file directory
# -------------------------------------------
filein.logdir = args.outdir + param.gsr + "_" + param.seedIDname + "/logs/"
isExist = os.path.exists(filein.logdir)
if not isExist:
    os.makedirs(filein.logdir)
    
    
# -------------------------------------------
#          Setup logging variables
# -------------------------------------------

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    filename=filein.logdir + 'pycap_Splithalf_postcap_impact_of_motion.log',
                    filemode='w')
console = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)






# -------------------------------------------
#          Define functions
# -------------------------------------------








# -------------------------------------------
#          Main analysis starts here.
# -------------------------------------------


msg = "============================================"
logging.info(msg)
msg = "Start processing ..."
logging.info(msg)



# -------------------------------------------
# - Load Frame Displacement Data from all subjects
# -------------------------------------------

flag_scrubbed_all,motion_metric  = motion_qc(filein=filein, param=param)
motion_metric = pd.DataFrame(motion_metric)

msg = "motion_metric = " + str(motion_metric)
logging.info(msg)


motion_stat = motion_metric.apply(lambda x: (x <= 0.5).sum())

# Cout low/high motion_framenum and proportion
motion_stat = motion_stat.to_frame()
motion_stat.columns = ['lowmotion_framenum']
motion_stat['lowmotion_proportion'] = (motion_stat['lowmotion_framenum']/4400)*100

motion_stat['highmotion_framenum'] = 4400 - motion_stat['lowmotion_framenum']
motion_stat['highmotion_proportion'] = (4400 - motion_stat['lowmotion_framenum'])/4400*100



# Compute the mean of the columns "highmotion_framenum" and "highmotion_proportion"

meanFD = motion_metric.mean()
stdFD = motion_metric.std()
motion_stat['meanFD'] = meanFD
motion_stat['stdFD'] = stdFD
mean_over_rows = motion_stat["meanFD"].mean()
std_over_rows = motion_stat["meanFD"].std()

# Print the result
print("Mean over subjects in meanFD = ", mean_over_rows)
print("SD over subjects in meanFD = ", std_over_rows)


mean_highmotion_framenum = motion_stat['highmotion_framenum'].mean()
std_highmotion_framenum = motion_stat['highmotion_framenum'].std()


mean_highmotion_proportion = motion_stat['highmotion_proportion'].mean()
std_highmotion_proportion = motion_stat['highmotion_proportion'].std()

# Print the results
print("Mean of highmotion_framenum:", mean_highmotion_framenum)
print("Standard deviation of highmotion_framenum:", std_highmotion_framenum)

print("Mean of highmotion_proportion:", mean_highmotion_proportion)
print("Standard deviation of highmotion_proportion:", std_highmotion_proportion)

print("Mean of motion_metric DataFrame:", meanFD)
print("Standard deviation of motion_metric DataFrame:", stdFD)


# Create a new column based on the condition (> 220 frames were scrubbed) >5%
motion_stat['above5%scrub'] = np.where(motion_stat['highmotion_framenum'] > 220, 1, 0)

# Count the number of subjects where more than 220 frames were scrubbed
count_above_220_frames_scrubbed = (motion_stat['above5%scrub'] == 1).sum()

# Print the result
print("Number of subjects where more than 220 frames were scrubbed:", count_above_220_frames_scrubbed)

# Select the index of the rows where more than 220 frames were scrubbed
subjects_above_220_frames_scrubbed = motion_stat[motion_stat['above5%scrub'] == 1].index.tolist()

# Print the list of indices
print("Indices of subjects where more than 220 frames were scrubbed:", subjects_above_220_frames_scrubbed)

subjects_above_220_frames_scrubbed= pd.DataFrame(subjects_above_220_frames_scrubbed)
highmotionsub_listfilen='/gpfs/gibbs/pi/n3//Studies/CAP_Time_Analytics/results/Splithalf_postcap/highmotionsub_list.csv'
subjects_above_220_frames_scrubbed.to_csv(highmotionsub_listfilen)

print(motion_stat)


#----------------------------------------------
# compute motiom continuity (length of consecutive motion time-frames)
#----------------------------------------------
# Scrubing with FD>0.5 mm
scrub_motion = motion_metric.applymap(lambda x: 1 if x <= 0.5 else 0)
# print(scrub_motion)


motion_length_per_column = []

# Iterate over each column in the binary DataFrame
for i, column in enumerate(scrub_motion.columns):
    current_motion_length = 0
    motion_lengths = []

    for value in scrub_motion[column]:
        if value == 0:
            # Increment motion length if the value is 0
            current_motion_length += 1
        elif current_motion_length > 0:
            # If value is 1 and there was a previous segment of 0s, append the motion length and reset
            motion_lengths.append(current_motion_length)
            current_motion_length = 0

    # Append the last motion length if the segment ends with 0
    if current_motion_length > 0:
        motion_lengths.append(current_motion_length)

    # Append the motion lengths for the current column to the list
    motion_length_per_column.append(motion_lengths)

    # Compute mean and standard deviation of motion lengths for the current column
    mean_motion_length = np.mean(motion_lengths)
    std_motion_length = np.std(motion_lengths)

    # Print mean and standard deviation
    print("Mean motion length for column", column, ":", mean_motion_length)
    print("Standard deviation of motion length for column", column, ":", std_motion_length)

    # Update the corresponding row in the motion_stat DataFrame with mean_motion_length
    motion_stat.at[column, 'mean_motion_length'] = mean_motion_length
    motion_stat.at[column, 'std_motion_length'] = std_motion_length
    motion_stat.at[column, 'motion_lengths'] = str(motion_lengths)

# Print the updated motion_stat DataFrame
print(motion_stat)


# Compute the mean of "mean_motion_length" across all subjects
mean_mean_motion_length = motion_stat['mean_motion_length'].mean()
std_mean_motion_length = motion_stat['mean_motion_length'].std()

# Print the result
print("Mean of mean motion length across all subjects:", mean_mean_motion_length)
print("Standard deviation of mean motion length across all subjects:", std_mean_motion_length)

highmotionsub_statfilen='/gpfs/gibbs/pi/n3//Studies/CAP_Time_Analytics/results/Splithalf_postcap/highmotionsub_stat.csv'
motion_stat.to_csv(highmotionsub_statfilen)


# - Notify job completion
msg = "========== The jobs are all completed. =========="
logging.info(msg)


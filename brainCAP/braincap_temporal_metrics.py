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
import math
import h5py
import os
import shutil
import random
import sklearn.model_selection
import numpy as np
import argparse
import itertools
import pandas as pd
import logging
from braincap_functions.loaddata import *
from braincap_functions.frameselection import *
from braincap_functions.gen import *
from braincap_functions.datasplit import *
import braincap_functions.exceptions as exceptions
import braincap_functions.utils as utils
import time
from scipy import stats
from itertools import groupby
from operator import itemgetter


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
        raise argparse.ArgumentTypeError(f"{path} must be a local path from the specified sessions_folder!")


parser = argparse.ArgumentParser()
parser.add_argument("--analysis_folder", type=dir_path, help="Output directory path")
parser.add_argument("--permutations", type=int, default=1, help="Range of permutations to run, default 1.")
parser.add_argument("--sessions_list", required=True, help="Path to list of sessions", type=file_path)
parser.add_argument("--overwrite", type=str, default="no", help='Whether to overwrite existing data')
parser.add_argument("--log_path", default='./prep_run_hcp.log', help='Path to output log', required=False)
parser.add_argument("--tag", default="", help="Tag for saving files, useful for doing multiple analyses in the same folder (Optional).")
#parser.add_argument("--bold_type", default=None, help="BOLD data type (CIFTI/NIFTI), if not supplied will use file extention")


args = parser.parse_args()  # Read arguments from command line

if args.tag != "":
    args.tag += "_"
overwrite = args.overwrite.lower()# == "yes"

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    filename=args.log_path,
                    filemode='w')
console = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

logging.info("BrainCAP Temporal Metrics Start")
#Wait for a moment so log tracking can keep up
time.sleep(1)

class Param:
    pass

param = Param()
# if not args.bold_type:
#     param.bold_type = utils.get_bold_type(args.bold_path)
# else:
#     param.bold_type = args.bold_type
param.tag = args.tag
param.overwrite = overwrite

class FileIn:
    pass


filein = FileIn()
filein.sublistfull, filein.groupsall = parse_sfile(args.sessions_list)
logging.info("Total number of subjects in the study (before split-half if used) = " + str(len(filein.sublistfull)))

basis_dir = os.path.join(args.analysis_folder, f"basis_CAPs")

#one basis CAP file per split

#Individual files per permutation if using. Reordered to use basis CAPs
metrics_ind_allcap_allperm = pd.DataFrame() #ultimate output variable to save

for perm_i in range(args.permutations):
    perm = perm_i + 1
    logging.info("==========")
    logging.info(f"Perm{perm}")
    logging.info("==========")

    for sp in [1, 2]:
        if sp == 1:
            param.spdatatag = "split1"
        elif sp == 2:
            param.spdatatag = "split2"

        logging.info("--------------------" + f"Perm{perm}" + " - " + param.spdatatag+ "--------------------")
        logging.info("Read data from " + param.spdatatag + "...")    
        #basis_path = os.path.join(basis_dir, f"{args.tag}basis_CAP_split{sp}.hdf5")
        #f = h5py.File(basis_path, 'r')
        #basis_data = np.array(f['basis_CAP'])
        #logging.info(param.spdatatag + ": Basis CAP shape " + str(basis_data.shape))

        reorder_path = os.path.join(basis_dir, f"{args.tag}reordered_clustering_results_perm{perm}_split{sp}.hdf5")
        r_f = h5py.File(reorder_path, 'r')
        r_clmean = np.array(r_f["cluster_means"]) # Reorded cluster means (3, 718)
        r_labels = np.array(r_f["cluster_labels"]) # Reorded cluster labels (215000,)
        #Subject IDs are saved as indices of the sessions_list using utils.id2index
        #   Same with Groups
        #index2id takes these indices and returns the relevant subject ID from the list
        r_sublabels = utils.index2id(np.array(r_f["sublabel_all"]), filein.sublistfull) #Reorded subject labels (215000,)
        r_grouplabels = utils.index2id(np.array(r_f["grouplabel"]), filein.groupsall) #Reorded group labels
        
        # QC
        n_cap = len(np.unique(r_labels))
        splitlist=np.unique(r_sublabels)
        n_sub = len(splitlist)
        totTR = len(r_labels)
        logging.info(param.spdatatag + ": Total number of clusters = " + str(n_cap))
        logging.info(param.spdatatag + ": Total number of subjects = " + str(n_sub))
        logging.info(param.spdatatag + ": Total number of time-points (all subjects) = " + str(totTR))
        if np.shape(r_labels) == np.shape(r_sublabels) == np.shape(r_grouplabels):
            logging.info("QC PASSED: All variables have the same number of time-points.")
        else:
            logging.info("QC FAILED: Shape mismatch detected.")
        
        # add columns
        metrics_ind_allcap = pd.DataFrame(splitlist, columns = ['subID'])
        metrics_ind_allcap["perm"] = perm_i
        metrics_ind_allcap["split"] = sp
        
        #------------------------------------------------------    
        # Compute CAP fractional occupancy for each individual
        #------------------------------------------------------
        logging.info("Computing CAP temporal measures (FA, mDT, vDT) for each individual...")        
        
        total_frames_check = 0    
        for i in range(n_cap):
    
            fa_ind = np.empty((0, 1), int)
            mDT_ind = []
            vDT_ind = []
            
            # Collect timeframes belonging to a CAP state for FA calculation
            idx = np.where(r_labels == i)[0]
            total_frames_check += len(idx)
            logging.info("-------------------------------")
            logging.info("CAP " + str(i) + ": a total of " + str(len(idx)) + " timeframes.")
            capstate = r_sublabels[idx] # array of subID from all frames belonging to this CAP
            
            for j in range(len(splitlist)):
                subj_id = splitlist[j]
                
                # Compute Total Accupancy of this CAP for each subject
                sidx = np.where(capstate == subj_id)[0]
                d = np.array([[len(sidx)]])
                d = d / totTR * 100 # Divide by the total number of frames 
                fa_ind = np.append(fa_ind, d, axis=0)
                
                
                # Compute mDT and vDT of this CAP for each subject
                sidx = np.where(r_sublabels == splitlist[j])[0] #index of timepoints that are from this subject
                cap_seq = r_labels[sidx] #array of cluster labels from this subject
                cap_mask = (cap_seq == i).astype(int)  # 1 if the frame belongs to CAP i, else 0
                
                logging.info("     Subject " + subj_id + " has " + str(len(cap_seq)) + " time-frames with CAP labels.")
                is_continuous = np.all(np.diff(sidx) == 1)
                if is_continuous:
                    logging.info("     - QC PASSED: sidx contains a continuous sequence of indices.")
                else:
                    logging.warning("     - QC FAILED: sidx does NOT contain a continuous sequence.")
                    logging.warning("     - Discontinuities at positions: %s", np.where(np.diff(sidx) != 1)[0])
                logging.info("     - Fractional Accupanry (CAP %d) %f", i, d)
                logging.info("     - cap_seq (first 30 time-frames): %s", cap_seq[:30])
                logging.info("     - CAP %d (first 30 time-frames): %s", i, cap_mask[:30])
                
                diff = np.diff(np.concatenate(([0], cap_mask, [0])))
                starts = np.where(diff == 1)[0]
                ends = np.where(diff == -1)[0]
                dwell_times = ends - starts  # lengths of consecutive CAP i segments
                logging.info("     - Dwell times (CAP %d) (first 10 segments) %s", i, dwell_times[:10])
                
                if len(dwell_times) > 0:
                    mdt = np.mean(dwell_times)
                    vdt = np.std(dwell_times)
                else:
                    mdt = 0.0
                    vdt = 0.0
                mDT_ind = np.append(mDT_ind, mdt)
                vDT_ind = np.append(vDT_ind, vdt)
                
            
            # Convert Total Accupancy to Fractional Accupancy    
            fa_ind = fa_ind.reshape(fa_ind.shape[0],)
            
        
            # Save individualmetrics for this CAP
            metrics_ind_tmp = pd.DataFrame({
                f"FA{i}": fa_ind,
                f"mDT{i}": mDT_ind,
                f"vDT{i}": vDT_ind
            })
            metrics_ind_allcap = pd.concat([metrics_ind_allcap, metrics_ind_tmp], axis=1)
            
            
        metrics_ind_allcap_allperm = pd.concat([metrics_ind_allcap_allperm, metrics_ind_allcap], axis=0, ignore_index=True)
        if total_frames_check == totTR:
            logging.info(f"QC PASSED: Sum of all CAP timeframes ({total_frames_check}) matches total time-points ({totTR}).")
        else:
            logging.warning(f"QC FAILED: Sum of all CAP timeframes ({total_frames_check}) does NOT match total time-points ({totTR}).")


msg= "\nTable: Individual CAP metrics : \n" + str(metrics_ind_allcap_allperm)
logging.info(msg)


metrics_ind_allcap_allperm_avg = metrics_ind_allcap_allperm.drop(columns=['perm', 'split']).groupby('subID', as_index=False).mean(numeric_only=True)
msg= "\nTable: Individual CAP metrics (averaged over permutations and splits) : \n" + str(metrics_ind_allcap_allperm_avg)
logging.info(msg)


# SAVE OUTPUT
out_dir = os.path.join(args.analysis_folder, "temporal_metrics") 
os.makedirs(out_dir, exist_ok=True)
fig_dir = os.path.join(out_dir, "figures") #For figures from this step
os.makedirs(fig_dir, exist_ok=True)
out_path = os.path.join(out_dir, f"{args.tag}temporal_metrics.hdf5") #If saving file per split, change to f"{args.tag}temporal_metrics_split{sp}.hdf5"

#f = h5py.File(out_path, 'w')
#f.create_dataset("grouplabel", (metrics_ind_allcap_allperm.shape[0],), dtype='int', data=metrics_ind_allcap_allperm)
#f.close()

logging.info("Saved data in " + out_path)
logging.info("Completed.")

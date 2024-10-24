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
from pycap_functions.pycap_loaddata import *
from pycap_functions.pycap_frameselection import *
from pycap_functions.pycap_gen import *
from pycap_functions.pycap_datasplit import *
import pycap_functions.pycap_exceptions as exceptions
import pycap_functions.pycap_utils as utils
import time


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
parser.add_argument("--scrubbing", type=str, help="Use scrugging or not (y/n)")
parser.add_argument("--save_image", type=str, default='no', help="Save CAP images or not (y/n)")
parser.add_argument("--k_method", default='silhouette', type=str, help="(sse/silhouette)")
parser.add_argument("-ev", "--event_combine", type=str, help="(average/interserction/union)")
parser.add_argument("-et", "--event_type", type=str, help="activation/deactivation/both")
parser.add_argument("--gsr", type=str, default="y", help="(y/n)")
parser.add_argument("--sessions_folder", type=dir_path, help="Home directory path")
parser.add_argument("--bold_path", type=local_path, help="Path to datafile inside session directory")
parser.add_argument("--analysis_folder", type=dir_path, help="Output directory path")
#parser.add_argument("--split", help="Which specific split(s) to run. If multiple, must be a pipe '|' seperated list.")
parser.add_argument("--n_splits", type=int, default=1, help="Range of splits to run, default 1. Will not be used if '--split' is specified.")
#In wrapper script, derived from k range
#parser.add_argument("--n_k", type=int, help="Number of clusters for k-means clustering")
#parser.add_argument("--max_iter", type=int, default=1000, help="Max iterations for k-means clustering")
parser.add_argument("--motion_type", type=str, help="(dvarsm,dvarsme,fd)")
parser.add_argument("--motion_path", type=str, help="Path to motion file inside session directory")
parser.add_argument("--seed_type", type=str, default="seedfree", help="(seedfree/seedbased), default 'seedfree'")
parser.add_argument("--seed_name", type=str, help="Seed name")
parser.add_argument("--seed_threshtype", type=str, help="(T/P)")
parser.add_argument("--seed_threshold", type=float, help="Signal threshold")
parser.add_argument("--sessions_list", required=True,
                    help="Path to list of sessions", type=file_path)
parser.add_argument("--subsplit_type", default='random', type=str, help="random/days, default 'random'")
parser.add_argument("--time_threshold", type=float, default=100, help="Random Time Signal threshold") #seedfree
#NEED FIX FOR DENSE 
parser.add_argument("--parc_file", type=file_path, required=False ,help="Path to parcellation template, required to save CAP image for parcellated data")
parser.add_argument("--motion_threshold", type=float, help="Motion threshold")
parser.add_argument("--display_motion", type=str,
                    help="Display motion parameter or not (y/n)")
#parser.add_argument("-step", "--step", type=str, help="Step to run (step1 or step2)")
parser.add_argument("--overwrite", type=str, default="no", help='Whether to overwrite existing data')
parser.add_argument("--log_path", default='./prep_run_hcp.log', help='Path to output log', required=False)
parser.add_argument("--mask", default=None, help="Brain mask, required for dense data and saving CAP image.")
parser.add_argument("--bold_type", default=None, help="BOLD data type (CIFTI/NIFTI), if not supplied will use file extention")
parser.add_argument("--cluster_args", type=str, required=True, help="Args for sklearn clustering in form 'key1=val1,key2=val2'. " \
                    "Must have key '_method', corresponding to a function in sklearn.clustering")
args = parser.parse_args()  # Read arguments from command line

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    filename=args.log_path,
                    filemode='w')
console = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

logging.info("PyCap Post Start")
#Wait for a moment so run_pycap.py log tracking can keep up
time.sleep(1)

# -------------------------------------------
#           Setup input parameters
# -------------------------------------------

overwrite = args.overwrite.lower() == "yes"

class Param:
    pass


param = Param()



# # - parameters for data selection
if args.gsr.lower() == "yes":
    param.gsr = "gsr"
#elif args.gsr.lower()  == "no":
else:
    param.gsr = "nogsr"

if 'ptseries' in args.bold_path:
    param.unit = 'p'
elif 'dtseries' in args.bold_path:
    param.unit = 'd'

param.mask = args.mask
if not args.bold_type:
    param.bold_type = utils.get_bold_type(args.bold_path)
else:
    param.bold_type = args.bold_type

# if param.unit == "d":
#     param.sdim = 91282
# elif param.unit == "p":
#     param.sdim = 718
# param.tdim = 4400
param.subsplit_type = args.subsplit_type


# - parameters for seed signal selection
param.seed_type = args.seed_type
if param.seed_type == "seedbased":
    utils.handle_args(args, ['seed_name','motion_type','motion_threshold','display_motion','event_combine','event_type'], 
                      'Prep', '--seed_type=seedbased')
    param.seedIDname = args.seed_name
    param.seedID = eval(param.seedIDname)
    param.event_combine = args.event_combine
    param.eventtype = args.event_type
    param.sig_thresholdtype = args.seed_threshtype
    param.sig_threshold = args.seed_threshold
#Defaults
if param.seed_type == "seedfree":
    param.seedIDname = param.seed_type
    param.time_threshold = args.time_threshold
    param.sig_thresholdtype = "P"

param.overwrite = overwrite

# # - parameters for motion scrubbing
if args.scrubbing.lower() == "yes":
    utils.handle_args(args, ['scrubbing','motion_type','motion_threshold','display_motion'], 
                      'Prep', '--scrubbing=yes')
    param.scrubbing = args.scrubbing
    param.motion_type = args.motion_type
    param.motion_threshold = args.motion_threshold  # dvarsmt=[3.0], dvarsmet=[1.6], fdr=[0.5]
    param.display_motion = args.display_motion
else:
    param.scrubbing = args.scrubbing
# param.n_dummy = args.ndummy
# #param.run_order = list(args.runorder)

# # - parameters for k-means clustering
param.cluster_args = utils.string2dict(args.cluster_args)
param.savecapimg = args.save_image


# -------------------------------------------
#              Setup input data
# -------------------------------------------


class FileIn:
    pass


filein = FileIn()
filein.sessions_folder = args.sessions_folder
filein.sublist = parse_slist(args.sessions_list)
filein.pscalar_filen = args.parc_file

filein.fname = args.bold_path
filein.motion_file = args.motion_path

# if '|' in args.split:
#     splits = args.split.split('|')
# else:
#     splits = [args.split]

for split_i in range(args.n_splits):
    #adjust to non-index count
    split = split_i + 1

    logging.info(f"Running split {split}")

    split_dir = os.path.join(args.analysis_folder, f"Ssplit{split}")
        
    filein.outpath = os.path.join(split_dir, f"{param.gsr}_{param.seedIDname}", 
                                    f"{param.sig_thresholdtype}{str(param.time_threshold)}/")

    filein.datadir = os.path.join(split_dir, f"{param.gsr}_{param.seedIDname}", 
                                    f"{param.sig_thresholdtype}{str(param.time_threshold)}", "session_data/")

    param.overwrite = args.overwrite
    # -------------------------------------------
    # - Population split-half list of subjects
    # -------------------------------------------

    test_sublist, training_sublist = subsplit(filein=filein, param=param)

    # -------------------------------------------
    # - Run the whole process for training and test datasets
    # -------------------------------------------
    for sp in [1, 2]:
        if sp == 1:
            param.spdatatag = "training_data"
            filein.sublist = training_sublist
        elif sp == 2:
            param.spdatatag = "test_data"
            filein.sublist = test_sublist

        msg = "============================================"
        logging.info(msg)
        msg = "Start processing " + param.spdatatag + "..."
        logging.info(msg)
        msg = "    >> np.unique(filein.sublist) : " + str(np.unique(filein.sublist))
        logging.info(msg)

        # Setup output directory
        # filein.analysis_folder = filein.outpath + param.spdatatag + "/"
        # isExist = os.path.exists(filein.analysis_folder)
        # if not isExist:
        #     os.makedirs(filein.analysis_folder)

        # -------------------------------------------
        # - Load a time by space data matrix from individual and temporally concatenate
        # -------------------------------------------

        data_all, sublabel_all = load_groupdata_wb_usesaved(filein=filein, param=param)
        msg = "    >> np.unique(sublabel_all) : " + str(np.unique(sublabel_all))
        logging.info(msg)

        # -------------------------------------------
        # - Frame-selection to find the moments of activation
        # -------------------------------------------

        if param.seed_type == "seedbased":
            # Reference: Liu and Duyn (2013), PNAS
            seeddata_all = load_groupdata_seed_usesaved(filein=filein, param=param)
            data_all_fsel, sublabel_all_fsel = frameselection_seed(
                inputdata=data_all, labeldata=sublabel_all, seeddata=seeddata_all, filein=filein, param=param)
        if param.seed_type == "seedfree":
            # Reference: Liu et al. (2013), Front. Syst. Neurosci.
            data_all_fsel, sublabel_all_fsel = frameselection_wb(
                inputdata=data_all, labeldata=sublabel_all, filein=filein, param=param)

        msg = "    >> np.unique(sublabel_all_fsel) : " + str(np.unique(sublabel_all_fsel))
        logging.info(msg)

        finalcluster2cap_any(inputdata=data_all_fsel, filein=filein, param=param)

        # -------------------------------------------
        # - Delete variable to save space
        # -------------------------------------------
        del data_all, sublabel_all, data_all_fsel, sublabel_all_fsel
        if param.seed_type == "seedbased":
            del seeddata_all

        msg = "\n"
        logging.info(msg)


    # - Notify job completion
    logging.info(f"--- STEP COMPLETE ---")

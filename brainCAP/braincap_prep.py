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
parser.add_argument("--gsr", type=str, default="y", help="(y/n)")
parser.add_argument("--sessions_folder", help="Home directory path")
parser.add_argument("--bold_path", type=local_path, help="Path to datafile inside session directory")
parser.add_argument("--analysis_folder", help="Output directory path")
parser.add_argument("--permutations", type=int, default=1, help="Number of permutations to run, default 1")
parser.add_argument("--motion_type", type=str, help="(dvarsm,dvarsme,fd)")
parser.add_argument("--motion_path", type=str, help="Path to motion file inside session directory")
parser.add_argument("--seed_args", type=str, default=None, help="dict consisting of seed arguments")
parser.add_argument("--sessions_list", required=True,
                    help="Path to list of sessions", type=file_path)
parser.add_argument("--time_threshold", type=float, default=100, help="Random Time Signal threshold")
parser.add_argument("--motion_threshold", type=float, help="Motion threshold")
parser.add_argument("--display_motion", type=str,
                    help="Display motion parameter or not (y/n)")
parser.add_argument("--overwrite", type=str, default="no", help='Whether to overwrite existing data')
parser.add_argument("--log_path", default='./prep_run_hcp.log', help='Path to output log', required=False)
parser.add_argument("--mask", default=None, help="Path to brain mask, recommended for dense data")
parser.add_argument("--bold_type", default=None, help="BOLD data type (CIFTI/NIFTI), if not supplied will use file extention")
parser.add_argument("--tag", default="", help="Tag for saving files, useful for doing multiple analyses in the same folder (Optional).")
args = parser.parse_args()  # Read arguments from command line

if args.tag != "":
    args.tag += "_"

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    filename=args.log_path,
                    filemode='w')
console = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

logging.info("BrainCAP Clusteing - Prep Start")
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
else:
    param.gsr = "nogsr"

if 'ptseries' in args.bold_path:
    param.unit = 'p'
elif 'dtseries' in args.bold_path:
    param.unit = 'd'
else:
    param.unit = 'n'
param.tag = args.tag
param.mask_file = args.mask
if not args.bold_type:
    param.bold_type = utils.get_bold_type(args.bold_path)
else:
    param.bold_type = args.bold_type

if not args.seed_args:
    param.seed_args == None
else:
    param.seed_args = utils.string2dict(args.seed_args)

param.overwrite = overwrite
param.randTthreshold = args.time_threshold
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

# -------------------------------------------
#              Setup input data
# -------------------------------------------


class FileIn:
    pass


filein = FileIn()
filein.sessions_folder = args.sessions_folder
filein.sublistfull, filein.groups = parse_sfile(args.sessions_list)
#filein.pscalar_filen = args.pscalarfile.name

filein.fname = args.bold_path
filein.motion_file = args.motion_path

for split_i in range(args.permutations):
    #adjust to non-index count
    split = split_i + 1

    split_dir = os.path.join(args.analysis_folder, f"perm{split}")
        
    filein.outpath = split_dir
    filein.datadir = os.path.join(split_dir, "data/")
    
    for path in [filein.outpath, filein.datadir]:

        try: 
            os.makedirs(path)
        except:
            logging.info(f"Output folder {path} already exists")

    # -------------------------------------------
    # - Population split-half list of subjects
    # -------------------------------------------

    split_2_sublist, split_1_sublist = subsplit(filein=filein, param=param)

    # -------------------------------------------
    # - Run the whole process for split_1 and split_2 datasets
    # -------------------------------------------
    for sp in [1, 2]:
        if sp == 1:
            param.spdatatag = "split1"
            filein.sublist = split_1_sublist
        elif sp == 2:
            param.spdatatag = "split2"
            filein.sublist = split_2_sublist

        msg = "============================================"
        logging.info(msg)
        msg = "Start processing " + param.spdatatag + "..."
        logging.info(msg)
        msg = "    >> np.unique(filein.sublist) : " + str(np.unique(filein.sublist))
        logging.info(msg)

        # -------------------------------------------
        # - Load a time by space data matrix from individual and temporally concatenate
        # -------------------------------------------

        data_all, sublabel_all, seeddata_all = create_groupdata(filein=filein, param=param)
        msg = "    >> session ids : " + str(np.unique(sublabel_all))
        logging.info(msg)

        # -------------------------------------------
        # - Frame-selection to find the moments of activation
        # -------------------------------------------

        data_all_fsel, sublabel_all_fsel = prep_scrubbed(
            inputdata=data_all, labeldata=sublabel_all, seeddata=seeddata_all, filein=filein, param=param)

        msg = "    >> np.unique(sublabel_all_fsel) : " + str(np.unique(sublabel_all_fsel))
        logging.info(msg)

        # -------------------------------------------
        # - Delete variable to save space
        # -------------------------------------------
        del data_all, sublabel_all, data_all_fsel, sublabel_all_fsel
        # if param.seed_based == "yes":
        #     del seeddata_all

        msg = "\n"
        logging.info(msg)


# - Notify job completion
logging.info(f"--- STEP COMPLETE ---")

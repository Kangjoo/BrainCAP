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

basis_dir = os.path.join(args.analysis_folder, f"basis_CAPs")

for sp in [1, 2]:
    if sp == 1:
        param.spdatatag = "split1"
    elif sp == 2:
        param.spdatatag = "split2"

    basis_path = os.path.join(basis_dir, f"{args.tag}basis_CAP_split{sp}.hdf5")
    f = h5py.File(basis_path, 'r')
    basis_data = np.array(f['basis_CAP'])
    logging.info("Basis CAP shape")
    logging.info(basis_data.shape)

    for perm_i in range(args.permutations):
        perm = perm_i + 1
        logging.info(f"Perm{perm}")
        reorder_path = os.path.join(basis_dir, f"{args.tag}reordered_clustering_results_perm{perm}_split{sp}.hdf5")
        r_f = h5py.File(reorder_path, 'r')
        r_clmean = np.array(r_f["cluster_means"])
        logging.info("Reorded cluster means shape")
        logging.info(r_clmean.shape)
        r_labels = np.array(r_f["cluster_labels"])
        logging.info("Reorded cluster labels shape")
        logging.info(r_labels.shape)
        r_sublabels = utils.index2id(np.array(r_f["sublabel_all"]), filein.sublistfull)
        logging.info("Reorded subject labels shape")
        logging.info(r_sublabels.shape)
        r_grouplabels = utils.index2id(np.array(r_f["grouplabel"]), filein.groupsall)
        logging.info("Reorded group labels shape")
        logging.info(r_grouplabels.shape)


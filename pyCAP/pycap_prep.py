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
from os.path import exists as file_exists
import random
import sklearn.model_selection
import numpy as np
import argparse
import itertools
import pandas as pd
import logging
from pycap_functions.pycap_loaddata_hcp import *
from pycap_functions.pycap_frameselection import *
from pycap_functions.pycap_gen import *
from pycap_functions.pycap_datasplit import *


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
parser.add_argument("--scrubbing", type=str, help="Use scrugging or not (y/n)")
#parser.add_argument("-ci", "--savecapimg", type=str, help="Save CAP images or not (y/n)")
#parser.add_argument("-d", "--ndummy", type=int, help="Number of dummy scans to remove")
#parser.add_argument("-e", "--kmethod", type=str, help="(sse/silhouette)")
#parser.add_argument("-ev", "--eventcombine", type=str, help="(average/interserction/union)")
#parser.add_argument("-et", "--eventtype", type=str, help="activation/deactivation/both")
parser.add_argument("--gsr", type=str, help="(y/n)")
parser.add_argument("--sessions_folder", type=dir_path, help="Home directory path")
parser.add_argument("--bold_path", type=local_path, help="Path to datafile inside session directory")
parser.add_argument("--analysis_folder", type=dir_path, help="Output directory path")
parser.add_argument("--n_splits", type=int, default=1, help="Number of splits to run, default 1")
#parser.add_argument("-dd", "--datadir", type=dir_path, help="Concatenated Data directory path")
#parser.add_argument("-k", "--ncluster", type=int, help="Number of clusters for k-means clustering")
#parser.add_argument("-kl", "--mink", type=int, help="Miminum k for k-means clustering")
#parser.add_argument("-ku", "--maxk", type=int, help="Maximum k for k-means clustering")
#parser.add_argument("-ki", "--maxiter", type=int, help="Iterations for k-menas clustering")
parser.add_argument("--motion_type", type=str, help="(dvarsm,dvarsme,fd)")
parser.add_argument("--motion_path", type=str, help="Path to motion file inside session directory")
#parser.add_argument("-s", "--seedtype", type=str, help="(seedfree/seedbased)")
#parser.add_argument("-si", "--seedname", type=str, help="Seed name")
#parser.add_argument("-sp", "--seedthreshtype", type=str, help="(T/P)")
#parser.add_argument("-st", "--seedthreshold", type=float, help="Signal threshold")
parser.add_argument("--sessions_list", required=True,
                    help="Path to list of sessions", type=file_path)
parser.add_argument("--subsplit_type", type=str, help="random/days")
parser.add_argument("--time_threshold", type=float, help="Random Time Signal threshold")
#parser.add_argument("-p", "--pscalarfilen", dest="pscalarfile", required=True,
#                    help="Pscalar filename", type=lambda f: open(f))
#parser.add_argument("-u", "--unit", type=str, help="(p/d)")  # parcel or dense
parser.add_argument("--motion_threshold", type=float, help="Motion threshold")
parser.add_argument("--display_motion", type=str,
                    help="Display motion parameter or not (y/n)")
#parser.add_argument("-step", "--step", type=str, help="Step to run (step1 or step2)")
parser.add_argument("--overwrite", type=str, default="no", help='Whether to overwrite existing data')
parser.add_argument("-l", "--log_path", default='./prep_run_hcp.log', help='Path to output log', required=False)
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

if 'ptseries' in args.bold_path:
    param.unit = 'p'
elif 'dtseries' in args.bold_path:
    param.unit = 'd'


if param.unit == "d":
    param.sdim = 91282
elif param.unit == "p":
    param.sdim = 718
param.tdim = 4400
param.subsplit_type = args.subsplit_type


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
    param.time_threshold = args.time_threshold

# - parameters for motion scrubbing
param.scrubbing = args.scrubbing
param.motion_type = args.motion_type
param.motion_threshold = args.motion_threshold  # dvarsmt=[3.0], dvarsmet=[1.6], fdr=[0.5]
param.display_motion = args.display_motion
param.n_dummy = args.ndummy
#param.run_order = list(args.runorder)

# - parameters for k-means clustering
param.kmean_k = args.ncluster
param.kmean_krange = [args.mink, args.maxk]
param.kmean_max_iter = args.maxiter
param.kmean_kmethod = args.kmethod
param.savecapimg = args.savecapimg


# -------------------------------------------
#              Setup input data
# -------------------------------------------


class FileIn:
    pass


filein = FileIn()
filein.sessions_folder = args.sessions_folder
#sl = pd.read_csv(args.sessions_list.name, header=0).values.tolist()
#filein.sublist = list(itertools.chain.from_iterable(sl))
filein.sublist = parse_slist(args.sessions_list)
filein.pscalar_filen = args.pscalarfile.name

filein.fname = args.bold_path
filein.motion_file = args.motion_path


# -------------------------------------------
#            Setup output directory
# -------------------------------------------

if param.seedtype == "seedbased":
    filein.outpath = args.analysis_folder + param.gsr + "_" + param.seedIDname + \
        "/" + param.sig_thresholdtype + str(param.sig_threshold) + "/"
    filein.datadir = args.datadir + param.gsr + "_" + param.seedIDname + \
        "/" + param.sig_thresholdtype + str(param.sig_threshold) + "/"
elif param.seedtype == "seedfree":
    filein.outpath = args.analysis_folder + param.gsr + "_" + \
        param.seedIDname + "/P" + str(param.time_threshold) + "/"
    filein.datadir = args.datadir + param.gsr + "_" + \
        param.seedIDname + "/P" + str(param.time_threshold) + "/"
filein.analysis_folder = filein.outpath

isExist = os.path.exists(filein.analysis_folder)
if not isExist:
    os.makedirs(filein.analysis_folder)
del isExist

isExist = os.path.exists(filein.datadir)
if not isExist:
    os.makedirs(filein.datadir)
del isExist


# -------------------------------------------
#          Setup logging variables
# -------------------------------------------

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    filename=args.log_path,
                    filemode='w')
console = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)


# -------------------------------------------
#          Main analysis starts here.
# -------------------------------------------

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
    filein.analysis_folder = filein.outpath + param.spdatatag + "/"
    isExist = os.path.exists(filein.analysis_folder)
    if not isExist:
        os.makedirs(filein.analysis_folder)

    # -------------------------------------------
    # - Load a time by space data matrix from individual and temporally concatenate
    # -------------------------------------------

    data_all, sublabel_all = load_hpc_groupdata_wb_usesaved(filein=filein, param=param)
    msg = "    >> np.unique(sublabel_all) : " + str(np.unique(sublabel_all))
    logging.info(msg)

    # -------------------------------------------
    # - Frame-selection to find the moments of activation
    # -------------------------------------------

    if param.seedtype == "seedbased":
        # Reference: Liu and Duyn (2013), PNAS
        seeddata_all = load_hpc_groupdata_seed_usesaved(filein=filein, param=param)
        data_all_fsel, sublabel_all_fsel = frameselection_seed(
            inputdata=data_all, labeldata=sublabel_all, seeddata=seeddata_all, filein=filein, param=param)
    elif param.seedtype == "seedfree":
        # Reference: Liu et al. (2013), Front. Syst. Neurosci.
        data_all_fsel, sublabel_all_fsel = frameselection_wb(
            inputdata=data_all, labeldata=sublabel_all, filein=filein, param=param)

    msg = "    >> np.unique(sublabel_all_fsel) : " + str(np.unique(sublabel_all_fsel))
    logging.info(msg)

    # -------------------------------------------
    # - Delete variable to save space
    # -------------------------------------------
    del data_all, sublabel_all, data_all_fsel, sublabel_all_fsel
    if param.seedtype == "seedbased":
        del seeddata_all

    msg = "\n"
    logging.info(msg)


# - Notify job completion
msg = f"========== PyCap `{args.step}` completed. =========="
logging.info(msg)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Kangjoo Lee (kangjoo.lee@yale.edu)
# Created Date: 12/17/2021
# Last Updated: 05/17/2022
# version ='0.0'
# ---------------------------------------------------------------------------

# =========================================================================
#                    --   Run pipeline template  --
#      Analysis of Co-Activation Patterns(CAP) in fMRI data (HCP-YA)
# =========================================================================

# Prerequisite libraries
#        NiBabel
#              https://nipy.org/nibabel/index.html
#              command: pip install nibabel
# (base) $ conda install matplotlib numpy pandas seaborn scikit-learn ipython h5py memory-profiler kneed nibabel

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
parser.add_argument("-c", "--scrubbing", type=str, help="Use scrugging or not (y/n)")
parser.add_argument("-ci", "--savecapimg", type=str, help="Save CAP images or not (y/n)")
parser.add_argument("-d", "--ndummy", type=int, help="Number of dummy scans to remove")
parser.add_argument("-e", "--kmethod", type=str, help="(sse/silhouette)")
parser.add_argument("-ev", "--eventcombine", type=str, help="(average/interserction/union)")
parser.add_argument("-et", "--eventtype", type=str, help="activation/deactivation/both")
parser.add_argument("-g", "--gsr", type=str, help="(y/n)")
parser.add_argument("-hd", "--homedir", type=dir_path, help="Home directory path")
parser.add_argument("-i", "--inputdata", type=str, help="Path to datafile inside session directory")
parser.add_argument("-od", "--outdir", type=dir_path, help="Output directory path")
parser.add_argument("-dd", "--datadir", type=dir_path, help="Concatenated Data directory path")
parser.add_argument("-k", "--ncluster", type=int, help="Number of clusters for k-means clustering")
parser.add_argument("-kl", "--mink", type=int, help="Miminum k for k-means clustering")
parser.add_argument("-ku", "--maxk", type=int, help="Maximum k for k-means clustering")
parser.add_argument("-ki", "--maxiter", type=int, help="Iterations for k-menas clustering")
parser.add_argument("-m", "--motiontype", type=str, help="(dvarsm,dvarsme,fd)")
#parser.add_argument("-mf", "--motionfile", type=str, help="(dvarsm,dvarsme,fd)")
parser.add_argument("-mf", "--motionfile", type=str, help="Path to motion file inside session directory")
parser.add_argument("-s", "--seedtype", type=str, help="(seedfree/seedbased)")
parser.add_argument("-si", "--seedname", type=str, help="Seed name")
parser.add_argument("-sp", "--seedthreshtype", type=str, help="(T/P)")
parser.add_argument("-st", "--seedthreshold", type=float, help="Signal threshold")
parser.add_argument("-sl", "--sublistfilen", required=True,
                    help="Path to list of sessions", type=file_path)
parser.add_argument("-ts", "--subsplittype", type=str, help="random/days")
parser.add_argument("-tt", "--randTthreshold", type=float, help="Random Time Signal threshold")
parser.add_argument("-p", "--pscalarfilen", dest="pscalarfile", required=True,
                    help="Pscalar filename", type=lambda f: open(f))
parser.add_argument("-u", "--unit", type=str, help="(p/d)")  # parcel or dense
parser.add_argument("-v", "--motionthreshold", type=float, help="Motion threshold")
parser.add_argument("-w", "--motiondisplay", type=str,
                    help="Display motio parameter or not (y/n)")
parser.add_argument("-step", "--step", type=str, help="Step to run (step1 or step2)")
parser.add_argument("-ow", "--overwrite", type=str, default="no", help='Whether to overwrite existing data')
args = parser.parse_args()  # Read arguments from command line


# -------------------------------------------
#      Define seed regions of interest
# -------------------------------------------

Thalamus_AD = [525, 526, 527, 528, 529, 538, 539, 540, 541]
Thalamus_CO = [290, 291, 312, 313]
Thalamus_DA = [350, 361]
Thalamus_DMN = [632, 633, 634, 647, 648, 649]
Thalamus_FP = [472, 494]
Thalamus_PM = [671, 686]
Thalamus_SM = [204, 217]
Thalamus_VIS1 = [39, 40, 41, 42, 43, 44, 66, 67, 68]
Thalamus_VIS2 = [138, 151]
# Thalamus_LAN = []
# Thalamus_VM = []
# Thalamus_OA = []

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

if 'ptseries' in args.inputdata:
    param.unit = 'p'
elif 'dtseries' in args.inputdata:
    param.unit = 'd'
else:
    param.unit = args.unit

if param.unit == "d":
    param.sdim = 91282
elif param.unit == "p":
    param.sdim = 718
param.tdim = 4800
param.subsplittype = args.subsplittype
param.motion_file = args.motionfile

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
filein.homedir = args.homedir
#sl = pd.read_csv(args.sublistfilen.name, header=0).values.tolist()
#filein.sublist = list(itertools.chain.from_iterable(sl))
filein.sublist = parse_slist(args.sublistfilen)
filein.pscalar_filen = args.pscalarfile.name

filein.fname = args.inputdata

#

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
#          Setup logging variables
# -------------------------------------------

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    filename=filein.outdir + 'pyCAP_run_hcp_' + args.step + '.log',
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
    filein.outdir = filein.outpath + param.spdatatag + "/"
    isExist = os.path.exists(filein.outdir)
    if not isExist:
        os.makedirs(filein.outdir)

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
    # - Apply k-means clustering on the concatenated data matrix
    # -------------------------------------------

    if args.step != 'prep':
        if args.step == "step1":
            clusterdata(inputdata=data_all_fsel, filein=filein, param=param)
        elif args.step == "step2":
            finalcluster2cap(inputdata=data_all_fsel, filein=filein, param=param)

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

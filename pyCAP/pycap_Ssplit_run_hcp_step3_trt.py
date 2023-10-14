#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Kangjoo Lee (kangjoo.lee@yale.edu)
# Created Date: 12/17/2021
# Last Updated: 11/01/2022
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
from pycap_functions.pycap_loaddata_hcp import *
from pycap_functions.pycap_frameselection import *
from pycap_functions.pycap_gen import *
from pycap_functions.pycap_datasplit import *
from pycap_functions.pycap_postcap_loaddata import *


def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--scrubbing", type=str, help="Use scrugging or not (y/n)")
parser.add_argument("-d", "--ndummy", type=int, help="Number of dummy scans to remove")
parser.add_argument("-pr", "--perm", type=int, help="Index of permutation")
parser.add_argument("-g", "--gsr", type=str, help="(y/n)")
parser.add_argument("-hd", "--homedir", type=dir_path, help="Home directory path")
parser.add_argument("-od", "--outdir", type=dir_path, help="Output directory path")
parser.add_argument("-dd", "--datadir", type=dir_path, help="Concatenated Data directory path")
parser.add_argument("-m", "--motiontype", type=str, help="(dvarsm,dvarsme,fd)")
parser.add_argument("-r", "--runorder", type=str, help="Order or runs to be concatenated")
parser.add_argument("-s", "--seedtype", type=str, help="(seedfree/seedbased)")
parser.add_argument("-sl", "--sublistfilen", dest="sublistfilen", required=True,
                    help="Subject list filename", type=lambda f: open(f))
parser.add_argument("-tt", "--randTthreshold", type=float, help="Random Time Signal threshold")
parser.add_argument("-p", "--pscalarfilen", dest="pscalarfile", required=True,
                    help="Pscalar filename", type=lambda f: open(f))
parser.add_argument("-u", "--unit", type=str, help="(p/d)")  # parcel or dense
parser.add_argument("-v", "--motionthreshold", type=float, help="Motion threshold")
parser.add_argument("-w", "--motiondisplay", type=str,
                    help="Display motio parameter or not (y/n)")
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


# - parameters for seed signal selection
param.seedtype = args.seedtype
if param.seedtype == "seedfree":
    param.seedIDname = param.seedtype
    param.randTthreshold = args.randTthreshold

# - parameters for motion scrubbing
param.scrubbing = args.scrubbing
param.motion_type = args.motiontype
param.motion_threshold = args.motionthreshold  # dvarsmt=[3.0], dvarsmet=[1.6], fdr=[0.5]
param.motion_display = args.motiondisplay
param.n_dummy = args.ndummy
param.run_order = list(args.runorder)
param.perm = args.perm


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
if param.seedtype == "seedfree":
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
                    filename=filein.outdir + 'output_step3.log',
                    filemode='w')
console = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)


# -------------------------------------------
#          Main analysis starts here.
# -------------------------------------------

test_sublist, training_sublist = subsplit(filein=filein, param=param)

# -------------------------------------------
# - Run the whole process for split 1 (named as "training") and split 2 ("test") datasets
# -------------------------------------------
for sp in [1,2]:
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

    # Setup output directory - for training/test separation
    filein.outdir = filein.outpath + param.spdatatag + "/"
    isExist = os.path.exists(filein.outdir)
    if not isExist:
        os.makedirs(filein.outdir)

    # -------------------------------------------
    # - Load a time by space data matrix from individual and temporally concatenate
    # -------------------------------------------

    data_all, sublabel_all, daylabel_all = load_hpc_groupdata_wb_daylabel(filein=filein, param=param)
    msg = "    >> np.unique(sublabel_all) : " + str(np.unique(sublabel_all))
    logging.info(msg)
    msg = "    >> daylabel_all " + str(daylabel_all)
    logging.info(msg)

    # -------------------------------------------
    # - Frame-selection to find the moments of activation
    # -------------------------------------------

    if param.seedtype == "seedfree":
        daylabel_all_fsel = frameselection_wb_daylabel(inputdata=data_all, daydata=daylabel_all, filein=filein, param=param)

    msg = "    >> np.unique(daylabel_all_fsel) : " + str(np.unique(daylabel_all_fsel))
    logging.info(msg)



    msg = "\n"
    logging.info(msg)


# - Notify job completion
msg = "========== The jobs are all completed. =========="
logging.info(msg)

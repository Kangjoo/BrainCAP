#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Kangjoo Lee (kangjoo.lee@yale.edu)
# Created Date: 04/05/2022
# Last Updated: 05/17/2022
# version ='0.0'
# ---------------------------------------------------------------------------

# =========================================================================
#                   --   Post CAP pipeline template  --
#      Analysis of Co-Activation Patterns(CAP) in fMRI data (HCP-YA)
# =========================================================================


# Imports
import h5py
import os
import glob
import numpy as np
import argparse
import itertools
import pandas as pd
import logging
from pycap_functions.pycap_createWBimages import *
from pycap_functions.pycap_postcap_loaddata import *
from pycap_functions.pycap_cap_timemetrics import *
from pycap_functions.pycap_gen import *
import ptitprince as pt
import matplotlib.collections as clt
import matplotlib.pyplot as plt
import seaborn as sns
# from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp


savefigs = True


def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")


parser = argparse.ArgumentParser()
parser.add_argument("-ci", "--savecapimg", type=str, help="Save CAP images or not (y/n)")
parser.add_argument("-g", "--gsr", type=str, help="(y/n)")
parser.add_argument("-hd", "--homedir", type=dir_path, help="Home directory path")
parser.add_argument("-od", "--outdir", type=dir_path, help="Output directory path")
parser.add_argument("-dd", "--datadirtag", type=str, help="CAP output directory path")
parser.add_argument("-pl", "--minperm", type=int, help="Miminum index of permutation")
parser.add_argument("-pu", "--maxperm", type=int, help="Maximum index of permutation")
parser.add_argument("-s", "--seedtype", type=str, help="(seedfree/seedbased)")
parser.add_argument("-si", "--seedname", type=str, help="Seed name")
parser.add_argument("-sp", "--seedthreshtype", type=str, help="(T/P)")
parser.add_argument("-st", "--seedthreshold", type=float, help="Signal threshold")
parser.add_argument("-sl", "--sublistfilen", dest="sublistfilen", required=True,
                    help="Subject list filename", type=lambda f: open(f))
parser.add_argument("-tt", "--standardTthreshold", type=float,
                    help="Standard Time Signal threshold")
parser.add_argument("-kr", "--basis_k_range", nargs='+',
                    type=int, help="basis k to be tested (range)")
parser.add_argument("-km", "--basismethod", type=str, help="Method to obtain basis CAP sets")
parser.add_argument("-p", "--pscalarfilen", dest="pscalarfile", required=True,
                    help="Pscalar filename", type=lambda f: open(f))
parser.add_argument("-u", "--unit", type=str, help="(p/d)")  # parcel or dense
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
    param.standardTthreshold = args.standardTthreshold

# - parameters for post-cap analysis
param.minperm = args.minperm
param.maxperm = args.maxperm
param.basis_k_range = args.basis_k_range
param.basismethod = args.basismethod

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


# -------------------------------------------
#            Setup output directory
# -------------------------------------------

param.datadirtag = args.datadirtag
if param.seedtype == "seedfree":
    filein.outdir = args.outdir + param.gsr + "_" + \
        param.seedIDname + "/P" + str(param.standardTthreshold) + "/"
isExist = os.path.exists(filein.outdir)
if not isExist:
    os.makedirs(filein.outdir)


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
                    filename=filein.logdir + 'output_postcap_step3.log',
                    filemode='w')
console = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)


# -------------------------------------------
#          Main analysis starts here.
# -------------------------------------------


param.splittype = "Ssplit"

msg = "============================================\n"
logging.info(msg)
msg = "Collect all permutation data from " + param.splittype + " dataset..\n"
logging.info(msg)

# ---------------------------------------------
# - Start collecting data across permutation
# ---------------------------------------------
training_clmean_all = np.empty([0, 718])
test_clmean_all = np.empty([0, 718])

for perm in range(param.minperm, param.maxperm+1):

    msg = param.splittype + " - Perm " + str(perm) + "\n"
    logging.info(msg)

    # -------------------------------------------
    # - Collect all CAP outputs from training/test datasets
    # -------------------------------------------

    if param.seedtype == "seedfree":
        filein.datapath = param.datadirtag + "/" + param.splittype + \
            str(perm) + "/" + param.gsr + "_" + param.seedIDname + \
            "/P" + str(param.standardTthreshold) + "/"
    training_data, test_data = load_capoutput(filein=filein, param=param)

    training_clmean_all = np.concatenate((training_clmean_all, training_data.clmean), axis=0)
    test_clmean_all = np.concatenate((test_clmean_all, test_data.clmean), axis=0)

    msg = "\n"
    logging.info(msg)
    msg = "Collected CAPs from training data: " + \
        str(training_clmean_all.shape) + ", test data: " + str(test_clmean_all.shape)
    logging.info(msg)

    # ---------------------------------------------
    # - End of collecting CAPs across permutation
    # ---------------------------------------------


# ------------------------------------------------
# - Generate a basis set from the concatenated CAP matrix
# ------------------------------------------------

for sp in [1, 2]:
    if sp == 1:
        spdatatag = "split_1"
        inputdata = training_clmean_all
    elif sp == 2:
        spdatatag = "split_2"
        inputdata = test_clmean_all

    msg = "\n"
    logging.info(msg)
    param.savecapimg = "y"

    # option 2: using the pre-selected k
    for param.kmean_k in param.basis_k_range:
        filein.outdir_keep = filein.outdir
        filein.outdir = filein.outdir + "/" + spdatatag + "/k" + str(param.kmean_k) + "/"
        isExist = os.path.exists(filein.outdir)
        if not isExist:
            os.makedirs(filein.outdir)
        if param.basismethod == "kmeans":
            param.kmean_kmethod = "silhouette"
            param.kmean_max_iter = 1000
            P = finalcluster2cap_noKestimate(inputdata=inputdata, filein=filein, param=param)
        elif param.basismethod == "pca":
            inputdata_pca = finalcluster2cap_pca(inputdata=inputdata, filein=filein, param=param)
        elif param.basismethod == "hac":
            inputdata_hac = finalcluster2cap_hac(inputdata=inputdata, filein=filein, param=param)
        filein.outdir = filein.outdir_keep

    # # option 1: using the estimation of k
    # for param.kmean_k in range(3, 8):
    #     P, score = clusterdata(inputdata=inputdata, filein=filein, param=param)
    # P = finalcluster2cap(inputdata=inputdata, filein=filein, param=param)


# - Notify job completion
msg = "\n========== The jobs are all completed. ==========\n"
logging.info(msg)

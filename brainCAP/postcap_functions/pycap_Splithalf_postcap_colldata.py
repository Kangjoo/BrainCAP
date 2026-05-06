#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# Created By  : Kangjoo Lee (kangjoo.lee@yale.edu)
# Last Updated: 08/06/2024
# -------------------------------------------------------------------------

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
parser.add_argument("-tt", "--randTthreshold", type=float, help="Random Time Signal threshold")
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
if param.seedtype == "seedbased":
    param.seedIDname = args.seedname
    param.seedID = eval(param.seedIDname)
    param.sig_thresholdtype = args.seedthreshtype
    param.sig_threshold = args.seedthreshold
elif param.seedtype == "seedfree":
    param.seedIDname = param.seedtype
    param.randTthreshold = args.randTthreshold

# - parameters for post-cap analysis
param.minperm = args.minperm
param.maxperm = args.maxperm


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
if param.seedtype == "seedbased":
    filein.outdir = args.outdir + param.gsr + "_" + param.seedIDname + \
        "/" + param.sig_thresholdtype + str(param.sig_threshold) + "/"
elif param.seedtype == "seedfree":
    filein.outdir = args.outdir + param.gsr + "_" + \
        param.seedIDname + "/P" + str(param.randTthreshold) + "/"
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
                    filename=filein.logdir + 'pycap_Splithalf_postcap_colldata.log',
                    filemode='w')
console = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)


# -------------------------------------------
#          Main analysis starts here.
# -------------------------------------------


splitlist = ["Ssplit"]


class Ssplit:
    pass


ssplit = Ssplit()

permseedID = [param.seedIDname] * (param.maxperm-param.minperm+1)
permseedID = np.array(permseedID)
if param.seedtype == "seedbased":
    permthresh = [str(int(param.sig_threshold))] * (param.maxperm-param.minperm+1)
elif param.seedtype == "seedfree":
    permthresh = [str(int(param.randTthreshold))] * (param.maxperm-param.minperm+1)
permthresh = np.array(permthresh)
n_perm = np.arange(param.minperm, param.maxperm+1)


for i in range(len(splitlist)):

    splittype = splitlist[i]
    param.splittype = splittype

    permsplit = [splittype] * (param.maxperm-param.minperm+1)
    permsplit = np.array(permsplit)

    msg = "============================================\n"
    logging.info(msg)
    msg = "Collect all permutation data from " + splittype + " dataset..\n"
    logging.info(msg)

    # ---------------------------------------------
    # - Start collecting data across permutation
    # ---------------------------------------------

    n_cap_training = np.empty((0, 1), int)  # Number of CAPS
    n_cap_test = np.empty((0, 1), int)
    fdim_clusterID_training = np.empty((0, 1), int)
    fdim_clusterID_test = np.empty((0, 1), int)
    fdim_subID_training = np.empty((0, 1), int)
    fdim_subID_test = np.empty((0, 1), int)

    for perm in range(param.minperm, param.maxperm+1):

        msg = splittype + " - Perm " + str(perm) + "\n"
        logging.info(msg)

        # -------------------------------------------
        # - Load CAP output from training/test datasets
        # -------------------------------------------

        if param.seedtype == "seedbased":
            filein.datapath = param.datadirtag + "/" + splittype + \
                str(perm) + "/" + param.gsr + "_" + param.seedIDname + "/" + \
                param.sig_thresholdtype + str(param.sig_threshold) + "/"
        elif param.seedtype == "seedfree":
            filein.datapath = param.datadirtag + "/" + splittype + \
                str(perm) + "/" + param.gsr + "_" + param.seedIDname + \
                "/P" + str(param.randTthreshold) + "/"
        training_data, test_data = load_capoutput(filein=filein, param=param)

        # -----------------------------------
        # - Load split info
        # -----------------------------------

        splitdata_filen = filein.datapath + "subsplit_datalist.hdf5"
        f = h5py.File(splitdata_filen, 'r')
        training_sublist_idx = f['training_sublist_idx']
        test_sublist_idx = f['test_sublist_idx']
        msg = str(training_sublist_idx)
        logging.info(msg)
        msg = str(test_sublist_idx)
        logging.info(msg)
        msg = filein.sublist
        logging.info(msg)

        test_data.splitlist = []
        for index in test_sublist_idx:
            test_data.splitlist.append(filein.sublist[index])
        training_data.splitlist = []
        for index in training_sublist_idx:
            training_data.splitlist.append(filein.sublist[index])

        del f, training_sublist_idx, test_sublist_idx, index

        msg = "============================================\n"
        logging.info(msg)
        msg = "We have split-half info data in " + splitdata_filen
        logging.info(msg)
        msg = "    #(training list) = " + str(len(training_data.splitlist)) + \
            ", #(test list) = " + str(len(test_data.splitlist))
        logging.info(msg)

        msg = "Training data list : " + str(training_data.splitlist)
        logging.info(msg)
        msg = "Test data list : " + str(test_data.splitlist)
        logging.info(msg)

        # -------------------------------------------
        # - Collect QC information
        # -------------------------------------------

        tmp = np.array([[len(training_data.frame_clusterID)]])
        fdim_clusterID_training = np.append(fdim_clusterID_training, tmp, axis=0)
        del tmp

        tmp = np.array([[len(training_data.frame_subID)]])
        fdim_subID_training = np.append(fdim_subID_training, tmp, axis=0)
        del tmp

        tmp = np.array([[len(test_data.frame_clusterID)]])
        fdim_clusterID_test = np.append(fdim_clusterID_test, tmp, axis=0)
        del tmp

        tmp = np.array([[len(test_data.frame_subID)]])
        fdim_subID_test = np.append(fdim_subID_test, tmp, axis=0)
        del tmp

        tmp = np.array([[training_data.clmean.shape[0]]])
        n_cap_training = np.append(n_cap_training, tmp, axis=0)
        del tmp

        tmp = np.array([[test_data.clmean.shape[0]]])
        n_cap_test = np.append(n_cap_test, tmp, axis=0)
        del tmp


    # ---------------------------------------------
    # - End of collecting data across permutation
    # ---------------------------------------------

    # --------------------------------------------------------
    # - Save QC info from all permutation for this splittype
    # --------------------------------------------------------

    fdim_clusterID_training = fdim_clusterID_training.reshape(fdim_clusterID_training.shape[0],)
    fdim_subID_training = fdim_subID_training.reshape(fdim_subID_training.shape[0],)
    fdim_clusterID_test = fdim_clusterID_test.reshape(fdim_clusterID_test.shape[0],)
    fdim_subID_test = fdim_subID_test.reshape(fdim_subID_test.shape[0],)
    n_cap_training = n_cap_training.reshape(n_cap_training.shape[0],)
    n_cap_test = n_cap_test.reshape(n_cap_test.shape[0],)

    # df for training data
    permhalf = ["training"] * n_cap_training.shape[0]
    permhalf = np.array(permhalf)
    df_warg = {'seedID': permseedID, 'seedthr': permthresh, 'split': permsplit, 'half': permhalf, 'nperm': n_perm,
               'fdim_clusterID': fdim_clusterID_training, 'fdim_subID': fdim_subID_training, 'n_cap': n_cap_training}
    col_warg = ['seedID', 'seedthr', 'split', 'half',
                'nperm', 'fdim_clusterID', 'fdim_subID', 'n_cap']
    df_training = pd.DataFrame(df_warg, columns=col_warg)
    del permhalf, df_warg, col_warg

    # df for test data
    permhalf = ["test"] * n_cap_training.shape[0]
    permhalf = np.array(permhalf)
    df_warg = {'seedID': permseedID, 'seedthr': permthresh, 'split': permsplit, 'half': permhalf, 'nperm': n_perm,
               'fdim_clusterID': fdim_clusterID_test, 'fdim_subID': fdim_subID_test, 'n_cap': n_cap_test}
    col_warg = ['seedID', 'seedthr', 'split', 'half',
                'nperm', 'fdim_clusterID', 'fdim_subID', 'n_cap']
    df_test = pd.DataFrame(df_warg, columns=col_warg)
    del permhalf, df_warg, col_warg

    # combine df for training and test data
    Ssplit_qc = pd.concat([df_training, df_test], axis=0, ignore_index=True)

# ----------------------------------------------------------
# - Combine Tsplit and Ssplit QC data and Save to a file
# ----------------------------------------------------------

Splitdata_qc = Ssplit_qc


Splitdata_qc.to_csv(filein.outdir + 'QC_output.csv')
msg = "Collected and saved QC data in: " + filein.outdir + 'QC_output.csv'
logging.info(msg)


# - Notify job completion
msg = "\n========== The jobs are all completed. ==========\n"
logging.info(msg)

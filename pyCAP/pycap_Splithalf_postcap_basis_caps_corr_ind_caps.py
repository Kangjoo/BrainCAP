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
from pycap_functions.pycap_loaddata_hcp import *
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
parser.add_argument("-ci", "--savecapimg", type=str, help="Save CAP images or not (y/n)")
parser.add_argument("-d", "--ndummy", type=int, help="Number of dummy scans to remove")
parser.add_argument("-e", "--kmethod", type=str, help="(sse/silhouette)")
parser.add_argument("-ev", "--eventcombine", type=str, help="(average/interserction/union)")
parser.add_argument("-et", "--eventtype", type=str, help="activation/deactivation/both")
parser.add_argument("-g", "--gsr", type=str, help="(y/n)")
parser.add_argument("-hd", "--homedir", type=dir_path, help="Home directory path")
parser.add_argument("-od", "--outdir", type=dir_path, help="Output directory path")
parser.add_argument("-dd", "--datadir", type=dir_path, help="Concatenated Data directory path")
parser.add_argument("-bd", "--basisoutdir", type=dir_path, help="Basis CAP output directory path")
parser.add_argument("-k", "--ncluster", type=int, help="Number of clusters for k-means clustering")
parser.add_argument("-kl", "--mink", type=int, help="Miminum k for k-means clustering")
parser.add_argument("-ku", "--maxk", type=int, help="Maximum k for k-means clustering")
parser.add_argument("-ki", "--maxiter", type=int, help="Iterations for k-menas clustering")
parser.add_argument("-m", "--motiontype", type=str, help="(dvarsm,dvarsme,fd)")
parser.add_argument("-r", "--runorder", type=str, help="Order or runs to be concatenated")
parser.add_argument("-s", "--seedtype", type=str, help="(seedfree/seedbased)")
parser.add_argument("-si", "--seedname", type=str, help="Seed name")
parser.add_argument("-sp", "--seedthreshtype", type=str, help="(T/P)")
parser.add_argument("-st", "--seedthreshold", type=float, help="Signal threshold")
parser.add_argument("-sl", "--sublistfilen", dest="sublistfilen", required=True,
                    help="Subject list filename", type=lambda f: open(f))
parser.add_argument("-ts", "--subsplittype", type=str, help="random/days")
parser.add_argument("-rtt", "--randTthreshold", type=float, help="Random Time Signal threshold")
parser.add_argument("-p", "--pscalarfilen", dest="pscalarfile", required=True,
                    help="Pscalar filename", type=lambda f: open(f))
parser.add_argument("-u", "--unit", type=str, help="(p/d)")  # parcel or dense
parser.add_argument("-v", "--motionthreshold", type=float, help="Motion threshold")
parser.add_argument("-w", "--motiondisplay", type=str,
                    help="Display motio parameter or not (y/n)")
parser.add_argument("-kr", "--basis_k_range", nargs='+',
                    type=int, help="basis k range")
parser.add_argument("-km", "--basismethod", type=str, help="Method to obtain basis CAP sets")    
parser.add_argument("-tt", "--standardTthreshold", type=float, help="Random Time Signal threshold")
parser.add_argument("-tr", "--randTthresholdrange", nargs='+',
                    type=int, help="Random timeframe threshold range")
                    
                    
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
    param.standardTthreshold = args.standardTthreshold
    param.randTthresholdrange = args.randTthresholdrange

# - parameters for motion scrubbing
param.scrubbing = args.scrubbing
param.motion_type = args.motiontype
param.motion_threshold = args.motionthreshold  # dvarsmt=[3.0], dvarsmet=[1.6], fdr=[0.5]
param.motion_display = args.motiondisplay
param.n_dummy = args.ndummy
param.run_order = list(args.runorder)

# - parameters for k-means clustering
param.kmean_k = args.ncluster
param.kmean_krange = [args.mink, args.maxk]
param.kmean_max_iter = args.maxiter
param.kmean_kmethod = args.kmethod
param.savecapimg = args.savecapimg

# - parameters for post-cap analysis
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
filein.basisoutdir = args.basisoutdir

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
                    filename=filein.logdir + 'pycap_Splithalf_postcap_basis_caps_corr_ind_caps.log',
                    filemode='w')
console = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)






# -------------------------------------------
#          Define functions
# -------------------------------------------


def corr2arrays(x, y):
    # X is an n x p input array
    # Y is an m x p input array 
    # output is an m x n correlation matrix
    mu_x = x.mean(1)
    mu_y = y.mean(1)
    n = x.shape[1]
    s_x = x.std(1, ddof=n - 1)
    s_y = y.std(1, ddof=n - 1)
    cov = np.dot(x,
                 y.T) - n * np.dot(mu_x[:, np.newaxis],
                                  mu_y[np.newaxis, :])
    R=cov / np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :])
    return R
    
    

def load_basiscaps_labels(spdatatag, filein, param, stdk):

    filein.basiscap_dir = filein.basisoutdir + "/" + param.gsr + "_" + param.seedtype + "/P" + str(param.standardTthreshold) + "/" + spdatatag + "/k" + str(stdk) + "/"    

    if param.basismethod == "hac":
        
        basiscap_clmean_filen = filein.basiscap_dir + \
            "FINAL_k" + str(stdk) + "_HACclustermean.hdf5"
        f = h5py.File(basiscap_clmean_filen, 'r')
        basisCAPs = f['clmean'][:]
        del f
      
    msg = "============================================\n"
    logging.info(msg)
    msg = "Load the basis " + str(stdk) + \
        " CAPs from P=" + str(param.standardTthreshold) + "% " + spdatatag
    logging.info(msg)
    msg = "    >> " + basiscap_clmean_filen + " " + str(basisCAPs.shape)
    logging.info(msg)

    return basisCAPs






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

    # Setup output directory
    filein.outdir = filein.outpath + param.spdatatag + "/"
    isExist = os.path.exists(filein.outdir)
    if not isExist:
        os.makedirs(filein.outdir)

    # # -------------------------------------------
    # # - Load a time by space data matrix from individual and temporally concatenate
    # # -------------------------------------------

    # data_all, sublabel_all = load_hpc_groupdata_wb_usesaved(filein=filein, param=param)
    # msg = "    >> np.unique(sublabel_all) : " + str(np.unique(sublabel_all))
    # logging.info(msg)

    # # -------------------------------------------
    # # - Frame-selection to find the moments of activation
    # # -------------------------------------------

    # if param.seedtype == "seedbased":
    #     # Reference: Liu and Duyn (2013), PNAS
    #     seeddata_all = load_hpc_groupdata_seed_usesaved(filein=filein, param=param)
    #     data_all_fsel, sublabel_all_fsel = frameselection_seed(
    #         inputdata=data_all, labeldata=sublabel_all, seeddata=seeddata_all, filein=filein, param=param)
    # elif param.seedtype == "seedfree":
    #     # Reference: Liu et al. (2013), Front. Syst. Neurosci.
    #     data_all_fsel, sublabel_all_fsel = frameselection_wb(
    #         inputdata=data_all, labeldata=sublabel_all, filein=filein, param=param)

    # msg = "    >> np.unique(sublabel_all_fsel) : " + str(np.unique(sublabel_all_fsel))
    # logging.info(msg)
    
    # msg = "    >> resultant data_all_fsel : " + str(data_all_fsel.shape) #e.g. Split 1- training: (727289, 718)
    # logging.info(msg)
    
    # msg = "    >> resultant sublabel_all_fsel : " + str(sublabel_all_fsel.shape)#e.g. Split 1- training: (727289,)
    # logging.info(msg)   
    

    # -------------------------------------------
    # - Load the basis CAP set
    # -------------------------------------------

    param.splittype = "Ssplit"
    if param.seedtype == "seedbased":
        pthrange = param.sig_threshold_range
    elif param.seedtype == "seedfree":
        pthrange = param.randTthresholdrange
    basiskrange = param.basis_k_range
    
    
    for pth in pthrange:
        
        if param.seedtype == "seedbased":
            param.sig_threshold = pth
        elif param.seedtype == "seedfree":
            param.randTthreshold = pth    
        
        
        
        for stdk in basiskrange:
    
    
            final_R = pd.DataFrame()
            
            
            for sp in [1, 2]:
                if sp == 1:
                    spdatatag = "split_1"
                elif sp == 2:
                    spdatatag = "split_2"
    
                # -------------- Load BASIS CAPs ----------------- #
                filein.basiscap_dir = filein.basisoutdir + "/" + param.gsr + "_" + param.seedtype + "/P" + str(param.standardTthreshold) + "/" + spdatatag + "/k" + str(stdk) + "/"   
                # msg = "    Load basis CAPs : " + filein.basiscap_dir
                # logging.info(msg)                
                # basisCAPs = load_basiscaps_labels(spdatatag=spdatatag, filein=filein, param=param, stdk=stdk)
            
                # msg = "    Compute correlation between individual CAPs and basis CAPs... "
                # logging.info(msg)                 
                # basiscap_indcap_R = corr2arrays(x=data_all_fsel,y=basisCAPs)
                # msg = "basiscap_indcap_R = " + str(basiscap_indcap_R.shape)
                # logging.info(msg)
                
                # basiscap_indcap_R = pd.DataFrame(basiscap_indcap_R)
                
                # Rfilen = filein.basiscap_dir + 'basiscap_indcap_R.csv'
                # basiscap_indcap_R.to_csv(Rfilen, index=False, header=False)
                
                # del basiscap_indcap_R
                

                Rfilen = filein.basiscap_dir + 'basiscap_indcap_R.csv'
                basiscap_indcap_R = pd.read_csv(Rfilen)
                msg = Rfilen
                logging.info(msg)                
                
                msg = "basiscap_indcap_R " + str(basiscap_indcap_R.shape)
                logging.info(msg)
                msg = "basiscap_indcap_R " + str(basiscap_indcap_R)
                logging.info(msg)
                
                # basiscap_indcap_R_abs = basiscap_indcap_R.abs()
                # basiscap_indcap_R_max = basiscap_indcap_R_abs.max(axis=1)
                basiscap_indcap_R_max = basiscap_indcap_R.max(axis=1)
                msg = "basiscap_indcap_R maximum" + str(basiscap_indcap_R_max)
                logging.info(msg)      
                
                basiscap_indcap_R_max = pd.DataFrame(basiscap_indcap_R_max, columns=['Max'])
                final_R = pd.concat([final_R, basiscap_indcap_R_max], axis=0, ignore_index=True)
                
                
            final_R.reset_index(drop=True, inplace=True)    
            msg = "basiscap_indcap_R maximum" + str(final_R.shape)
            logging.info(msg)     
            mean_max = final_R["Max"].mean()
            std_max = final_R["Max"].std()

            plt.figure(figsize=(7, 7))
            sns.histplot(data=final_R, x="Max",bins=30,kde=True)
            # sns.kdeplot(final_R["Max"],fill=True)
            plt.xlabel('R')
            plt.ylabel('Density')
            plt.title('R\nMean: {:.2f}, Std: {:.2f}'.format(mean_max, std_max))
            # plt.show()
            # plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
            savefilen = filein.basiscap_dir + 'basiscap_indcap_R.png'    
            plt.savefig(savefilen, bbox_inches='tight')
            plt.close()  
            
                
    # # -------------------------------------------
    # # - Delete variable to save space
    # # -------------------------------------------
    # del data_all, sublabel_all, data_all_fsel, sublabel_all_fsel
    # if param.seedtype == "seedbased":
    #     del seeddata_all

    msg = "\n"
    logging.info(msg)


# - Notify job completion
msg = "========== The jobs are all completed. =========="
logging.info(msg)


# # - Delete data directory
# try:
#     shutil.rmtree(filein.datadir)
# except OSError as e:
#     print("Error: %s : %s" % (filein.datadir, e.strerror))

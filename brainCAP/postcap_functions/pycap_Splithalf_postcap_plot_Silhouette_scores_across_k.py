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
import nibabel as nib
import numpy as np
import argparse
import itertools
import pandas as pd
import scipy as sp
from scipy import stats
import logging
import matplotlib.collections as clt
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from kneed import KneeLocator


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
parser.add_argument("-sr", "--seedthresholdrange", nargs='+',
                    type=int, help="Signal threshold range")
parser.add_argument("-sl", "--sublistfilen", dest="sublistfilen", required=True,
                    help="Subject list filename", type=lambda f: open(f))
parser.add_argument("-tr", "--randTthresholdrange", nargs='+',
                    type=int, help="Random timeframe threshold range")
parser.add_argument("-kr", "--basis_k_range", nargs='+',
                    type=int, help="basis k range")                    
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
    param.sig_threshold_range = args.seedthresholdrange
elif param.seedtype == "seedfree":
    param.seedIDname = param.seedtype
    param.randTthresholdrange = args.randTthresholdrange

# - parameters for post-cap analysis
param.minperm = args.minperm
param.maxperm = args.maxperm
param.basis_k_range = args.basis_k_range


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
filein.outdir = args.outdir + param.gsr + "_" + param.seedIDname + "/"
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
                    filename=filein.logdir + 'pycap_Splithalf_postcap_plot_Silhouette_scores_across_k.log',
                    filemode='w')
console = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)



# -------------------------------------------
#          Define functions.
# -------------------------------------------


# -------------------------------------------
#          Define classes.
# -------------------------------------------




class opt:
    pass


opt = opt()


# -------------------------------------------
#          Main analysis starts here.
# -------------------------------------------


param.splittype = "Ssplit"

if param.seedtype == "seedbased":
    pthrange = param.sig_threshold_range
elif param.seedtype == "seedfree":
    pthrange = param.randTthresholdrange





for pth in pthrange:

    if param.seedtype == "seedbased":
        param.sig_threshold = pth
    elif param.seedtype == "seedfree":
        param.randTthreshold = pth
        
    # Create an empty list to store all silhouette scores
    silhouette_score_all = []
    
    # Iterate over permutations, split types, and k values
    for perm in np.arange(param.minperm, param.maxperm + 1):
        for splittype in ["training", "test"]:
            for k in np.arange(2, 16):
                fpath = param.datadirtag + '/Ssplit' + str(perm) + '/gsr_seedfree/P100.0/' + splittype + '_data/kmeans_k' + str(k) + '_silhouette_score.csv'
                print(fpath)
                    
                # Read the CSV file
                score = np.loadtxt(fpath)
                
                # Create a dictionary with the current data
                temp_dict = {
                    'perm': perm,
                    'k': k,
                    'splittype': splittype,
                    'silhouette_score': score
                }
                
                # Append the dictionary to silhouette_score_all list
                silhouette_score_all.append(temp_dict)
    
    # Convert silhouette_score_all list of dictionaries to a dataframe
    silhouette_score_all = pd.DataFrame(silhouette_score_all)
    
    print(silhouette_score_all)





    
    # Create a plot
    plt.figure(figsize=(7, 7))
    
    # Iterate over permutations and split types and draw lines
    for perm in np.arange(param.minperm, param.maxperm + 1):
        for splittype in ["training", "test"]:
            data = silhouette_score_all[(silhouette_score_all['perm'] == perm) & (silhouette_score_all['splittype'] == splittype)]
            print(data)
            plt.plot(data['k'], data['silhouette_score'], marker='o', markersize=5, color='black')

            # ---------------------------------------
            # -        Determine optimal k
            # ---------------------------------------
            kl = KneeLocator(data['k'], data['silhouette_score'], curve="convex", direction="decreasing")
            final_k = kl.elbow
            print("Perm " + str(perm) + ", " + splittype + ": data:" + str(data) + " >>> optimal k = " + str(final_k))

            # Highlight the value of final_k on the plot
            if final_k is not None:
                final_k_score = data[data['k'] == final_k]['silhouette_score'].iloc[0]
                plt.plot(final_k, final_k_score, marker='o', markersize=10, color='red')
            
    # Add labels and legend
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores across k')
    plt.legend()
    
    # Save the plot as a .png file
    savefilen=filein.outdir + 'CAP_KmeansClustering_silhouette_scores_across_k_permMIN' + str(param.minperm) + 'MAX' + str(param.maxperm) + '_bothsplits.png'
    plt.savefig(savefilen)
    print("Saved in " + savefilen)


# - Notify job completion
msg = "\n========== The jobs are all completed. ==========\n"
logging.info(msg)











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
from braincap_functions.plots import plot_histogram
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
parser.add_argument("--save_image", type=str, default='no', help="Save CAP images or not (y/n)")
parser.add_argument("--sessions_folder", type=dir_path, help="Home directory path")
parser.add_argument("--analysis_folder", type=dir_path, help="Output directory path")
parser.add_argument("--permutations", type=int, default=1, help="Range of permutations to run, default 1.")
parser.add_argument("--sessions_list", required=True,
                    help="Path to list of sessions", type=file_path)
parser.add_argument("--parc_file", type=file_path, required=False ,help="Path to parcellation template, required to save CAP image for parcellated data")
parser.add_argument("--overwrite", type=str, default="no", help='Whether to overwrite existing data')
parser.add_argument("--log_path", default='./prep_run_hcp.log', help='Path to output log', required=False)
parser.add_argument("--mask", default=None, help="Brain mask, required for dense data and saving CAP image.")
parser.add_argument("--bold_type", default=None, help="BOLD data type (CIFTI/NIFTI), if not supplied will use file extention")
parser.add_argument("--cluster_args", type=str, required=True, help="Args for sklearn clustering in form 'key1=val1,key2=val2'. " \
                    "Must have key '_method', corresponding to a function in sklearn.clustering")
parser.add_argument("--tag", default="", help="Tag for saving files, useful for doing multiple analyses in the same folder (Optional).")
#Basis Cap params
#Whether to generate a histogram of cluster-values across permutations
parser.add_argument("--save_stats", default="yes")
#Picking which k value to use. If int, then means manual selection of k
#If automatic, uses mode
parser.add_argument("--cluster_selection", default="automatic", help="")
#Whether to use all permutations, or only those where the chosen k value is optimum
parser.add_argument("--use_all", default="yes")
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

logging.info("BrainCAP Clustering - Post Start")
time.sleep(1)

# -------------------------------------------
#           Setup input parameters
# -------------------------------------------

overwrite = args.overwrite.lower()# == "yes"

class Param:
    pass


param = Param()

param.mask = args.mask
if not args.bold_type:
    param.bold_type = utils.get_bold_type(args.bold_path)
else:
    param.bold_type = args.bold_type

param.tag = args.tag

param.overwrite = overwrite

# # - parameters for clustering
param.cluster_args = utils.string2dict(args.cluster_args)
param.savecapimg = args.save_image


# -------------------------------------------
#              Setup input data
# -------------------------------------------


class FileIn:
    pass


filein = FileIn()
filein.sessions_folder = args.sessions_folder
filein.sublistfull, filein.groups = parse_sfile(args.sessions_list)
filein.pscalar_filen = args.parc_file

#c_vals = np.zeros((args.permutations,2))
c_vals = [[],[]]

#Determine optimum clusters
for split_i in range(args.permutations):
    #adjust to non-index count
    split = split_i + 1

    logging.info(f"Running permutation {split}")

    split_dir = os.path.join(args.analysis_folder, f"perm{split}")
        
    filein.outpath = split_dir
    filein.datadir = os.path.join(split_dir, "data/")
    param.overwrite = args.overwrite
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

        #c_vals[split_i][sp-1] = determine_clusters(filein, param)
        c_vals[sp-1].append(determine_clusters(filein, param))

c_vals = np.vstack(c_vals).T


if args.save_stats == "yes":
    logging.info("Creating cluster value histograms..")

    fig_path = os.path.join(args.analysis_folder, 'figures')
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    cluster_args = param.cluster_args.copy()
    cluster_method = cluster_args.pop('_method', None)
    c_var = cluster_args.pop('_variable', None)
    c_range = [min(cluster_args[c_var]), max(cluster_args[c_var])]
    for sp in [1,2]:
        plot_histogram(c_vals[:,sp-1], c_range, os.path.join(fig_path,f"{c_var}_histogram_split{sp}.png"))

if args.cluster_selection == "automatic":
    final_c = stats.mode(c_vals,axis=0)[0]
else:
    final_c = [int(args.cluster_selection)] * 2

clusters_all = [[],[]]
mean_all = [[],[]]

for split_i in range(args.permutations):
    #adjust to non-index count
    split = split_i + 1

    logging.info(f"Running permutation {split}")

    split_dir = os.path.join(args.analysis_folder, f"perm{split}")
        
    filein.outpath = split_dir
    filein.datadir = os.path.join(split_dir, "data/")
    param.overwrite = args.overwrite
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

        if final_c[sp-1] != c_vals[split_i][sp-1]:
            logging.info(f"Warning: ideal cluster value is for perm{split} {param.spdatatag} is {c_vals[split_i][sp-1]}, but selected value is {final_c[sp-1]}")
            if args.use_all == "yes":
                logging.info(f"--use_all set to 'yes', using perm{split} anyway!")
            else:
                logging.info(f"--use_all set to '{args.use_all}', skipping...")
                continue


        msg = "============================================"
        logging.info(msg)
        msg = "Start processing " + param.spdatatag + "..."
        logging.info(msg)
        msg = "    >> np.unique(filein.sublist)a : " + str(np.unique(filein.sublist))
        logging.info(msg)

        data_all_fsel, sublabel_all_fsel = load_groupdata(filein, param)

        msg = "    >> np.unique(sublabel_all_fsel) : " + str(np.unique(sublabel_all_fsel))
        logging.info(msg)

        clusters, mean_data = finalcluster2cap_any(inputdata=data_all_fsel, filein=filein, param=param, final_k=final_c[sp-1])

        clusters_all[sp-1].append(clusters)
        mean_all[sp-1].append(mean_data)
        # -------------------------------------------
        # - Delete variable to save space
        # -------------------------------------------
        del data_all_fsel, sublabel_all_fsel

        msg = "\n"
        logging.info(msg)

basis_out = os.path.join(args.analysis_folder, 'basis_CAPs')
if not os.path.exists(basis_out):
    os.makedirs(basis_out)

for sp in [1, 2]:
    basis_cap = np.mean(mean_all[sp-1], axis=1)
    f_path = os.path.join(basis_out, f"basis_CAP_split{sp}{args.tag}_clustermean.hdf5")
    if os.path.exists(f_path):
        logging.info(f"WARNING: Outputfile {f_path} exists")
        if args.overwrite == "yes":
            logging.info("Overwrite set to 'yes', existing file will be replaced...")
            os.remove(f_path)
        else:
            logging.info("Overwrite set to 'no', skipping...")
            continue

    f = h5py.File(f_path, "w")
    f.create_dataset("basis", dtype="float32", data=basis_cap)
    f.close()

# - Notify job completion
logging.info(f"--- STEP COMPLETE ---")

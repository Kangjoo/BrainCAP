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
parser.add_argument("--sessions_list", required=True, help="Path to list of sessions", type=file_path)
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

splits = 2 #Add as param

# -------------------------------------------
#              Setup input data
# -------------------------------------------


class FileIn:
    pass


filein = FileIn()
filein.sessions_folder = args.sessions_folder
filein.sublistfull, filein.groupsall = parse_sfile(args.sessions_list)
filein.pscalar_filen = args.parc_file
filein.fig_folder = os.path.join(args.analysis_folder, 'figures')
if not os.path.exists(filein.fig_folder):
    os.makedirs(filein.fig_folder)
#c_vals = np.zeros((args.permutations,2))
c_vals = [[],[]]

#For plotting
#cluster_args = param.cluster_args.copy()a
cluster_method = param.cluster_args['_method']
c_var = param.cluster_args['_variable']
c_range = [min(param.cluster_args[c_var]), max(param.cluster_args[c_var])]

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

    # -------------------------------------------
    # - Run the whole process for split_1 and split_2 datasets
    # -------------------------------------------
    for sp in [1, 2]:
        if sp == 1:
            param.spdatatag = "split1"
        elif sp == 2:
            param.spdatatag = "split2"
        #c_vals[split_i][sp-1] = determine_clusters(filein, param)
        score_all, final_k = determine_clusters(filein, param)

        plot_scree(param.cluster_args[c_var], score_all, c_var, final_k, os.path.join(filein.fig_folder, f"{c_var}_scree_perm{split}_{sp}.png"))
    
        c_vals[sp-1].append(final_k)

c_vals = np.vstack(c_vals).T


if args.save_stats == "yes":
    logging.info("Creating cluster value histograms..")
    for sp in [1,2]:
        plot_histogram(c_vals[:,sp-1], c_range, os.path.join(filein.fig_folder,f"{c_var}_distribution_split{sp}.png"))

if args.cluster_selection == "automatic":
    final_c = stats.mode(c_vals,axis=0)[0]
    for c in final_c:
        if c != final_c[0]:
            raise exceptions.StepError("BrainCAP automatic cluster determination",\
                                       "Different optimal clustering values were found between splits",
                                       f"Specify the final clustering value manually, recommended one of: {final_c}")
else:
    final_c = [int(args.cluster_selection)] * 2

first=True
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

    #split_2_sublist, split_1_sublist = subsplit(filein=filein, param=param)

    # -------------------------------------------
    # - Run the whole process for split_1 and split_2 datasets
    # -------------------------------------------
    
    for sp in [1, 2]:
        #For filling np array
        index_c = 0
        if sp == 1:
            param.spdatatag = "split1"
            #filein.sublist = split_1_sublist
        elif sp == 2:
            param.spdatatag = "split2"
            #filein.sublist = split_2_sublist

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

        data_all_fsel, sublabel_all_fsel, group_all_fsel = load_groupdata(filein, param)

        clusters, mean_data = finalcluster2cap_any(inputdata=data_all_fsel, filein=filein, param=param, final_k=final_c[sp-1])

        #Create arrays to hold data, built here because nclusters (mean_data.shape[0]) is unknown
        if first:
            n_clusters = mean_data.shape[0]
            cluster_means = np.empty((splits, n_clusters*args.permutations, mean_data.shape[1])) 
            #cluster_labels = np.empty((splits,clusters.shape[0]*args.permutations))
            first=False

        cluster_means[sp-1,n_clusters*split_i:n_clusters*split,:] = mean_data
        #cluster_labels[sp-1,index_t:index_t+clusters.shape[0]] = clusters

        
        # -------------------------------------------
        # - Delete variable to save space
        # -------------------------------------------
        del data_all_fsel, sublabel_all_fsel, group_all_fsel, mean_data, clusters

        msg = "\n"
        logging.info(msg)

basis_out = os.path.join(args.analysis_folder, 'basis_CAPs')
if not os.path.exists(basis_out):
    os.makedirs(basis_out)

for sp in [1, 2]:
    if sp == 1:
        param.spdatatag = "split1"
    elif sp == 2:
        param.spdatatag = "split2"
    
    f_path = os.path.join(basis_out, f"{args.tag}basis_CAP_split{sp}.hdf5")
    if os.path.exists(f_path):
        logging.info(f"WARNING: Outputfile {f_path} exists")
        if args.overwrite == "yes":
            logging.info("Overwrite set to 'yes', existing file will be replaced...")
            os.remove(f_path)
        else:
            logging.info("Overwrite set to 'no', skipping...")
            continue

    

    basis_labels, basis_cap = create_basis_CAP(cluster_means[sp-1], n_clusters)
    f = h5py.File(f_path, "w")
    f.create_dataset("basis_CAP", dtype="float32", data=basis_cap)
    f.create_dataset("basis_CAP_labels", dtype="int", data=basis_labels)
    f.close()

    #Re-order
    for perm_i in range(args.permutations):
        perm = perm_i + 1
        split_dir = os.path.join(args.analysis_folder, f"perm{perm}")
        #filein.outpath = split_dir
        filein.datadir = os.path.join(split_dir, "data/")

        clmean = cluster_means[sp-1,perm_i*n_clusters:perm*n_clusters,:]
        corr_matrix = np.corrcoef(clmean,basis_cap)[0:n_clusters,n_clusters:]
        sorted_R, sortcap_match = reorder_R(corr_matrix, 'test')

        c_results_f = os.path.join(split_dir,f"{param.tag}clustering_results_{param.spdatatag}.hdf5")
        labels = np.asarray(h5py.File(c_results_f, "r")["cluster_labels"])

        reordered_clmean = np.empty_like(clmean)
        reordered_labels = np.empty_like(labels)

        for i, c in enumerate(sortcap_match):
            reordered_clmean[i] = reordered_clmean[c]
            reordered_labels[np.where(labels==i)] = c

        data_all_fsel, sublabel_all_fsel, group_all_fsel = load_groupdata(filein, param)

        f_path = os.path.join(basis_out, f"{args.tag}reordered_clustering_results_perm{perm}_split{sp}.hdf5")
        if os.path.exists(f_path):
            logging.info(f"WARNING: Outputfile {f_path} exists")
            if args.overwrite == "yes":
                logging.info("Overwrite set to 'yes', existing file will be replaced...")
                os.remove(f_path)
            else:
                logging.info("Overwrite set to 'no', skipping...")
                continue

        f = h5py.File(f_path, "w")
        f.create_dataset("cluster_means", dtype="float32", data=reordered_clmean)
        f.create_dataset("cluster_labels", dtype="int", data=reordered_labels)
        f.create_dataset("sublabel_all", (sublabel_all_fsel.shape[0],), dtype='int', data=utils.id2index(sublabel_all_fsel,filein.sublistfull))
        f.create_dataset("grouplabel", (group_all_fsel.shape[0],), dtype='int', data=utils.id2index(group_all_fsel,filein.groupsall))
        f.close()

# - Notify job completion
logging.info(f"--- STEP COMPLETE ---")

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
from pycap_functions.pycap_gen import *
import ptitprince as pt
import matplotlib.collections as clt
import matplotlib.pyplot as plt
import seaborn as sns
rpal=sns.color_palette("coolwarm", as_cmap=True)


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
parser.add_argument("-tt", "--standardTthreshold", type=float, help="Random Time Signal threshold")
parser.add_argument("-tr", "--randTthresholdrange", nargs='+',
                    type=int, help="Random timeframe threshold range")
parser.add_argument("-kr", "--basis_k_range", nargs='+',
                    type=int, help="basis k range")
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
if param.seedtype == "seedbased":
    param.seedIDname = args.seedname
    param.seedID = eval(param.seedIDname)
    param.sig_thresholdtype = args.seedthreshtype
    param.sig_threshold = args.seedthreshold
elif param.seedtype == "seedfree":
    param.seedIDname = param.seedtype
    param.standardTthreshold = args.standardTthreshold
    param.randTthresholdrange = args.randTthresholdrange

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
if param.seedtype == "seedbased":
    filein.outdir = args.outdir + param.gsr + "_" + param.seedIDname + \
        "/"
elif param.seedtype == "seedfree":
    filein.outdir = args.outdir + param.gsr + "_" + \
        param.seedIDname + "/"
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
                    filename=filein.logdir + 'pycap_Splithalf_postcap_basis_45caps_margdist.log',
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



def reorder_R(R, savefilen,saveimgflag):
    # R is an n x m correlation matrx (e.g. # of test caps x # of basis caps)
    # the goal is to re-order(re-label) n rows (test caps)
    # to assign a cap index for test cap according to the similarity with basis caps
    # Output: an n x m correlation matrix

    if (R.shape[0] == R.shape[1]):

        msg = "Re-label (re-order) rows of the correlation matrix by sorting test caps (rows) using spatial similarity to basis CAPs."
        logging.info(msg)
        sortcap_match = np.zeros((R.shape[1],))
        for basis_c in np.arange(R.shape[1]):
            sortcap = R[:, basis_c].argsort()
            sortcap = sortcap[::-1]
            sortcap_match[basis_c] = sortcap[0]
        sortcap_match = np.int_(sortcap_match)
        del sortcap

        if np.array_equal(np.unique(sortcap_match), np.unique(np.arange(R.shape[1]))):
            msg = "All test caps are sorted using spatial correlation (r) with basis CAPs."
            logging.info(msg)
        else:
            msg = "There is one or more test caps that are assigned to more than 1 basis CAPs."
            logging.info(msg)

            # ovl_testcapID: Indeces of test caps that are assigned to more than 1 basis CAP
            match, counts = np.unique(sortcap_match, return_counts=True)
            if len(np.where(counts > 1)[0] == 1):
                ovl_testcapID = match[np.where(counts > 1)[0]]

            # Do following: if found one test CAP assigned to more than 1 basis CAP
            # Goal: to compare actual r value of this test CAP with two basis CAPs
            # and assign this test CAP to the basis CAP with higher r
            if len(ovl_testcapID == 1):
                # ovl_basiscapID: Indices of basis CAPs that have assigned to the same test cap
                ovl_basiscapID = np.where(sortcap_match == ovl_testcapID)[0]
                r_tocompare = R[ovl_testcapID, ovl_basiscapID]
                keep_idx = ovl_basiscapID[np.where(r_tocompare == max(r_tocompare))[0]]
                replace_idx = ovl_basiscapID[np.where(r_tocompare == min(r_tocompare))[0]]

                msg = "R(testcap" + str(ovl_testcapID) + ", basiscap" + \
                    str(ovl_basiscapID) + ")=" + str(r_tocompare)
                logging.info(msg)
                msg = "basiscap " + str(keep_idx) + \
                    "should be matched to testcap " + str(ovl_testcapID) + "."
                logging.info(msg)
                msg = "basiscap " + str(replace_idx) + " should be matched to other testcap."
                logging.info(msg)

                missing_idx = np.array(list(set(np.arange(R.shape[1])).difference(match)))
                msg = "Found a test cap without assignment : " + str(missing_idx)
                logging.info(msg)

                sortcap_match[replace_idx] = missing_idx

        if np.array_equal(np.unique(sortcap_match), np.unique(np.arange(R.shape[1]))):
            sorted_R = R[sortcap_match]
            f, ax = plt.subplots(figsize=(4, 8))
            plt.subplot(211)
            ax = sns.heatmap(R, annot=True, linewidths=.5, vmin=-1, vmax=1, cmap=rpal)
            plt.xlabel('basis CAPs')
            plt.ylabel('test CAPs')
            plt.subplot(212)
            ax = sns.heatmap(sorted_R, annot=True, linewidths=.5, vmin=-1, vmax=1, cmap=rpal)
            plt.xlabel('basis CAPs')
            plt.ylabel('Re-ordered test CAPs (new label)')
            # plt.show()
            if saveimgflag == 1:
                plt.savefig(savefilen, bbox_inches='tight')
                msg = "Saved " + savefilen
                logging.info(msg)
        else:
            msg = "Cannot save " + savefilen + ": caps were not matched. " + str(sortcap_match)
            logging.info(msg)

    elif (R.shape[0] < R.shape[1]):

        msg = "Re-label (re-order) rows of the correlation matrix by sorting test caps (rows) using spatial similarity to basis CAPs."
        logging.info(msg)
        sortcap_match = np.zeros((R.shape[0],))
        for est_c in np.arange(R.shape[0]):
            sortcap = R[est_c, :].argsort()
            sortcap = sortcap[::-1]
            sortcap_match[est_c] = sortcap[0]
        sortcap_match = np.int_(sortcap_match)
        del sortcap

        if np.array_equal(np.unique(sortcap_match), np.unique(np.arange(R.shape[0]))):
            sorted_R = np.zeros((R.shape))
            for j in np.arange(R.shape[0]):
                idx = sortcap_match[j]
                sorted_R[idx, :] = R[j, :]
            f, ax = plt.subplots(figsize=(4, 8))
            plt.subplot(211)
            ax = sns.heatmap(R, annot=True, linewidths=.5, vmin=-1, vmax=1, cmap=rpal)
            plt.xlabel('basis CAPs')
            plt.ylabel('test CAPs')
            plt.subplot(212)
            ax = sns.heatmap(sorted_R, annot=True, linewidths=.5, vmin=-1, vmax=1, cmap=rpal)
            plt.xlabel('basis CAPs')
            plt.ylabel('Re-ordered test CAPs (new label)')
            # plt.show()
            if saveimgflag == 1:
                plt.savefig(savefilen, bbox_inches='tight')
                msg = "Saved " + savefilen
                logging.info(msg)
        else:
            msg = "Cannot save " + savefilen + ": caps were not matched. " + str(sortcap_match)
            logging.info(msg)

    return sorted_R, sortcap_match


def compute_reliability_caps_margdist(spdatatag, filein, param, stdk, testk):

    filein.basiscap_dir = filein.outdir + "/P" + \
        str(param.standardTthreshold) + "/" + spdatatag + "/k" + str(stdk) + "/"

    if param.basismethod == "kmeans":
        basiscap_clmean_filen = filein.basiscap_dir + \
            "FINAL_k" + str(stdk) + "_Kmeansclustermean.hdf5"
        f = h5py.File(basiscap_clmean_filen, 'r')
        basisCAPs = f['clmean'][:]
        del f
    elif param.basismethod == "hac":
        basiscap_clmean_filen = filein.basiscap_dir + \
            "FINAL_k" + str(stdk) + "_HACclustermean.hdf5"
        f = h5py.File(basiscap_clmean_filen, 'r')
        basisCAPs = f['clmean'][:]
        del f
    elif param.basismethod == "pca":
        basiscap_clmean_filen = filein.basiscap_dir + "FINAL_k" + str(stdk) + "_PCA.hdf5"
        f = h5py.File(basiscap_clmean_filen, 'r')
        basisCAPs = f['inputdata_pca'][:]
        del f

    msg = "============================================\n"
    logging.info(msg)
    msg = "Load the basis " + str(stdk) + \
        " CAPs from P=" + str(param.standardTthreshold) + "% " + spdatatag
    logging.info(msg)
    msg = "    >> " + basiscap_clmean_filen + " " + str(basisCAPs.shape)
    logging.info(msg)
    
    # Compute spatial similarity between the basis CAPs
    basiscapR = corr2arrays(x=basisCAPs,y=basisCAPs)
    # ax = sns.heatmap(basiscapR, annot=True, linewidths=.5, vmin=-1, vmax=1)
    mask = np.zeros_like(basiscapR)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(7, 5))
        ax = sns.heatmap(basiscapR, mask=mask, annot=True, annot_kws={"size": 20},  fmt='.2f',linewidths=.5, vmin=-1, vmax=1, square=True, cmap=rpal)
    plt.xlabel('basis CAPs')
    plt.ylabel('basis CAPs')
    # plt.show()
    savefilen = filein.outdir + "btw_basiscap" + str(stdk) + "_corr_" + spdatatag + ".png"
    plt.savefig(savefilen, bbox_inches='tight')
    msg = "Saved " + savefilen
    logging.info(msg)

    # Collect correlation matrixces
    msg ="\n"
    logging.info(msg)
    msg = spdatatag + " : Collect " + str(testk) +"-by-" + str(stdk) + \
            " correlation matrices estimated between the " + str(stdk) + \
            " basis sets vs (k=" + str(testk) + ") solution CAPs among " + \
            str(param.maxperm) + " permutations."
    logging.info(msg)
    R_allperm = collect_r_basis2test(basisCAPs=basisCAPs, filein=filein, param=param, testk=testk)
    msg = spdatatag + " : Collected " + str(testk) +"-by-" + str(stdk) + \
            " correlation matrices estimated between the " + str(stdk)+ \
            " basis sets vs (k=" + str(testk) + ") solution CAPs from " + \
            str(R_allperm.shape[2]) + "/" + str(param.maxperm) + " permutations."
    logging.info(msg)   
    msg = spdatatag + " : R_allperm = " + str(R_allperm.shape) 
    logging.info(msg)
    msg ="\n"
    logging.info(msg)
    
    
    # draw marginal distribution plots for each basis CAP
    for basis_c in np.arange(stdk):
        margdata = R_allperm[:,basis_c,:]
        margdata = np.reshape(margdata,(R_allperm.shape[0],R_allperm.shape[2]))
        margdata = np.transpose(margdata)
        msg = spdatatag + " - CAP " + str(basis_c) + ": estimate marginal distribution from data " + str(margdata.shape) 
        logging.info(msg)
        
        r_colname='r(testcap,basiscap' + str(basis_c) + ')'
        col_warg = [r_colname, 'testcap_idx']
        df_basis2test = pd.DataFrame(columns = col_warg)
        for k in np.arange(testk):
            rdist=margdata[:,k]
            kstring='testcap' + str(k)
            permk = [kstring] * margdata.shape[0]
            permk = np.array(permk)
            df_warg = {r_colname: rdist, 'testcap_idx': permk}
            
            df_test = pd.DataFrame(df_warg, columns=col_warg)
            del permk, df_warg
            df_basis2test=pd.concat([df_basis2test, df_test], axis=0,ignore_index=True)           

        sns.displot(df_basis2test, x=r_colname, hue="testcap_idx", kind="kde", fill=True)
        plt.xlim((-1.2, 1.2))
        plt.xticks([-1, 0, 1])
        savefilen = filein.outdir + "margdist_basiscapID"+ str(basis_c) + "_testk" + str(testk) +"_basisset" + str(stdk) + "_" + spdatatag + ".png"
        plt.savefig(savefilen, bbox_inches='tight')   
        msg="Saved " + savefilen
        logging.info(msg)
        savefilen2 = filein.outdir + "margdist_basiscapID"+ str(basis_c) + "_testk" + str(testk) +"_basisset" + str(stdk) + "_" + spdatatag + ".csv"
        df_basis2test.to_csv(savefilen2)
        msg="Saved " + savefilen
        logging.info(msg)

    return R_allperm


def collect_r_basis2test(basisCAPs, filein, param, testk):
    
    R_allperm = np.zeros(shape=(basisCAPs.shape[0],testk,))
    reordered_cap = pd.DataFrame(columns=["perm","neworder"])
    
    for perm in range(param.minperm, param.maxperm+1):

        msg = "\n"
        logging.info(msg)
        msg = "Load CAPs from " + param.splittype + " with P= " + \
            str(param.randTthreshold) + "% dataset: " + str(perm)+"-th permutation:..\n"
        logging.info(msg)

        # -------------------------------------------
        # - Load CAPs
        # -------------------------------------------

        if param.seedtype == "seedbased":
            filein.datapath = param.datadirtag + "/" + param.splittype + \
                str(perm) + "/" + param.gsr + "_" + param.seedIDname + "/" + \
                param.sig_thresholdtype + str(param.sig_threshold) + ".0/"
        elif param.seedtype == "seedfree":
            filein.datapath = param.datadirtag + "/" + param.splittype + \
                str(perm) + "/" + param.gsr + "_" + param.seedIDname + \
                "/P" + str(param.randTthreshold) + ".0/"

        msg = filein.datapath
        logging.info(msg)
        training_data, test_data = load_capoutput(filein=filein, param=param)

        if spdatatag == "split_1":
            testcap = training_data.clmean
        elif spdatatag == "split_2":
            testcap = test_data.clmean

        msg = "\n"
        logging.info(msg)
        msg = ">> Select the " + spdatatag + " clmean = " + str(testcap.shape) + "\n"
        logging.info(msg)

        # -------------------------------------------
        # - Compute spatial correlation to basis CAPs
        # -------------------------------------------
        
        if testcap.shape[0] == testk:
            R = corr2arrays(x=testcap,y=basisCAPs)
            msg = str(R)
            logging.info(R)
            
            #-rank-order the correlation matrix to assign cap index to test caps for further analysis
            savefilen = filein.outdir + "sorted_testcaps_vs_basiscaps_R_" + spdatatag + "_perm" + str(perm) + "_basis" + str(basisCAPs.shape[0]) + "_testk" + str(testk) + ".png"
            if perm < 11:
                saveimgflag = 1
            else:
                saveimgflag = 0
            R,sortcap_match=reorder_R(R=R, savefilen=savefilen, saveimgflag=saveimgflag)
            
            reordered_cap.loc[len(reordered_cap.index)] = [perm, 1] 
            reordered_cap.at[len(reordered_cap.index)-1,"neworder"]=sortcap_match
            
            if np.sum(R_allperm)==0:
                R=np.reshape(R,(R.shape[0],R.shape[1],-1))
                msg=str(R.shape)
                logging.info(msg)
                R_allperm=R
            else:
                R_allperm = np.dstack([R_allperm,R])
                msg = "Collected the " + str(R.shape) + " correlation matrix from "+ str(perm) + "-th permutation: " + spdatatag + ". Size=" + str(R_allperm.shape) + ". "
                logging.info(msg)            
        else:
            msg= "Do not collect correlation matrix because not a k=" + str(testk) + " solution from the " + str(perm) + "-th permutation."
            logging.info(msg)
    
    msg = str(reordered_cap)
    logging.info(msg)        
    savefilen2 = filein.outdir + "sorted_testcaps_neworder_" + spdatatag + "_allperms_basis" + str(basisCAPs.shape[0]) + "_testk" + str(testk) + ".csv"     
    reordered_cap.to_csv(savefilen2)
        
    return R_allperm



def load_basiscaps(spdatatag, filein, param, stdk):

    filein.basiscap_dir = filein.outdir + "/P" + \
        str(param.standardTthreshold) + "/" + spdatatag + "/k" + str(stdk) + "/"

    if param.basismethod == "kmeans":
        basiscap_clmean_filen = filein.basiscap_dir + \
            "FINAL_k" + str(stdk) + "_Kmeansclustermean.hdf5"
        f = h5py.File(basiscap_clmean_filen, 'r')
        basisCAPs = f['clmean'][:]
        del f
    elif param.basismethod == "hac":
        basiscap_clmean_filen = filein.basiscap_dir + \
            "FINAL_k" + str(stdk) + "_HACclustermean.hdf5"
        f = h5py.File(basiscap_clmean_filen, 'r')
        basisCAPs = f['clmean'][:]
        del f
    elif param.basismethod == "pca":
        basiscap_clmean_filen = filein.basiscap_dir + "FINAL_k" + str(stdk) + "_PCA.hdf5"
        f = h5py.File(basiscap_clmean_filen, 'r')
        basisCAPs = f['inputdata_pca'][:]
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
param.splittype = "Ssplit"
if param.seedtype == "seedbased":
    pthrange = param.sig_threshold_range
elif param.seedtype == "seedfree":
    pthrange = param.randTthresholdrange
basiskrange = param.basis_k_range

for stdk in basiskrange:

    for sp in [1, 2]:
        if sp == 1:
            spdatatag = "split_1"
        elif sp == 2:
            spdatatag = "split_2"

        # ============================================================
        # Compare the standard CAPs to the CAPs from each threshold to test
        # ============================================================

        for pth in pthrange:

            if param.seedtype == "seedbased":
                param.sig_threshold = pth
            elif param.seedtype == "seedfree":
                param.randTthreshold = pth

            if stdk == 4:
                testk=stdk
                R_allperm = compute_reliability_caps_margdist(spdatatag=spdatatag, filein=filein, param=param, stdk=stdk, testk=testk)
            elif stdk == 5:
                for testk in [4, 5]:
                    R_allperm = compute_reliability_caps_margdist(spdatatag=spdatatag, filein=filein, param=param, stdk=stdk, testk=testk)
            
            del R_allperm
            
plt.close('all')


for sp in [1, 2]:
    if sp == 1:
        spdatatag = "split_1"
    elif sp == 2:
        spdatatag = "split_2"
    # ============================================================
    # Compute the spatial similarity between the 4-bCAPs and 5-bCAPs
    # ============================================================
        
    basis4CAPs = load_basiscaps(spdatatag=spdatatag, filein=filein, param=param, stdk=4)        
    basis5CAPs = load_basiscaps(spdatatag=spdatatag, filein=filein, param=param, stdk=5) 

    # Compute spatial similarity between the basis CAPs
    basiscapR = corr2arrays(x=basis4CAPs,y=basis5CAPs)
    ax = sns.heatmap(basiscapR, annot=True, annot_kws={"size": 20},  fmt='.2f',linewidths=.5, vmin=-1, vmax=1, square=True, cmap=rpal)
    plt.xlabel('5-CAP basis set')
    plt.ylabel('4-CAP basis set')
    # plt.show()
    savefilen = filein.outdir + "btw_45basiscap_corr_" + spdatatag + ".png"
    plt.savefig(savefilen, bbox_inches='tight')
    msg = "Saved " + savefilen
    logging.info(msg)
    plt.close()
 
    
plt.close('all')    

  
for stdk in basiskrange:
    # ============================================================
    # Compute the spatial similarity between the 4-bCAPs and 5-bCAPs
    # ============================================================
        
    basisCAPs_split1 = load_basiscaps(spdatatag="split_1", filein=filein, param=param, stdk=stdk)        
    basisCAPs_split2 = load_basiscaps(spdatatag="split_2", filein=filein, param=param, stdk=stdk) 

    # Compute spatial similarity between the basis CAPs
    basiscapR = corr2arrays(x=basisCAPs_split1,y=basisCAPs_split2)
    ax = sns.heatmap(basiscapR, annot=True, annot_kws={"size": 20},  fmt='.2f',linewidths=.5, vmin=-1, vmax=1, square=True, cmap=rpal)
    plt.xlabel(str(stdk) + '-CAP basis set - ' + "split2")
    plt.ylabel(str(stdk) + '-CAP basis set - ' + "split1")
    # plt.show()
    savefilen = filein.outdir + "btwsplit_" + str(stdk) +"basiscap_corr.png"
    plt.savefig(savefilen, bbox_inches='tight')
    msg = "Saved " + savefilen
    logging.info(msg)
    plt.close()
     
    
    

# - Notify job completion
msg = "\n========== The jobs are all completed. ==========\n"
logging.info(msg)

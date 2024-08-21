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
import scipy as sp
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
parser.add_argument("-ct", "--classTh", nargs='+', type=int, help="Subgroup classification threshold for z score")     
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
param.classTh = args.classTh

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
                    filename=filein.logdir + 'pycap_Splithalf_postcap_subset_ksolution.log',
                    filemode='w')
console = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)


# -------------------------------------------
#          Define functions
# -------------------------------------------


def getsplitlist_perm(permlist, param, filein, halftype):

    permsub_training = []
    permsub_test = []

    if (len(permlist) > 0):
        for perm in permlist:

            if param.seedtype == "seedbased":
                filein.datapath = param.datadirtag + "/" + param.splittype + \
                    str(perm) + "/" + param.gsr + "_" + param.seedIDname + "/" + \
                    param.sig_thresholdtype + str(param.sig_threshold) + ".0/"
            elif param.seedtype == "seedfree":
                filein.datapath = param.datadirtag + "/" + param.splittype + \
                    str(perm) + "/" + param.gsr + "_" + param.seedIDname + \
                    "/P" + str(param.randTthreshold) + ".0/"

            # -----------------------------------
            # - Load subject-based split info
            # -----------------------------------

            splitdata_filen = filein.datapath + "subsplit_datalist.hdf5"
            f = h5py.File(splitdata_filen, 'r')
            training_sublist_idx = f['training_sublist_idx']
            test_sublist_idx = f['test_sublist_idx']

            if halftype == "training":

                training_data_splitlist = []
                for index in training_sublist_idx:
                    training_data_splitlist.append(filein.sublist[index])
                permsub_training = permsub_training + training_data_splitlist
                # msg = "    >> " + splitdata_filen + " " + \
                #     "permsub_training (n=" + str(len(training_data_splitlist)) + \
                #     "): " + str(training_data_splitlist)
                # logging.info(msg)
                permsub = permsub_training

            elif halftype == "test":

                test_data_splitlist = []
                for index in test_sublist_idx:
                    test_data_splitlist.append(filein.sublist[index])
                permsub_test = permsub_test + test_data_splitlist
                # msg = "    >> " + splitdata_filen + " " + \
                #     "permsub_test (n=" + str(len(test_data_splitlist)) + "): " + \
                #     str(test_data_splitlist)
                # logging.info(msg)
                permsub = permsub_test

        msg = "    >> Concatenated subject IDs in " + halftype + " permsub n = " + str(len(permsub))
        logging.info(msg)
    elif (len(permlist) == 0):
        permsub = [103515] #just a random ID
        msg = "    >> Concatenated subject IDs in " + halftype + " permsub n = None."
        logging.info(msg)

    return permsub


# -------------------------------------------
#          Main analysis starts here.
# -------------------------------------------
param.splittype = "Ssplit"
if param.seedtype == "seedbased":
    pthrange = param.sig_threshold_range
elif param.seedtype == "seedfree":
    pthrange = param.randTthresholdrange
basiskrange = param.basis_k_range


# -------------------------------------------
#            Setup colormap
# -------------------------------------------

n_perm = np.arange(param.minperm, param.maxperm+1)
Ncolors = n_perm.shape[0]
colormap = plt.cm.PRGn
Ncolors = min(colormap.N, Ncolors)
mapcolors = [colormap(int(x*colormap.N/Ncolors)) for x in range(Ncolors)]

if param.seedtype == "seedbased":
    pthrange = param.sig_threshold_range
elif param.seedtype == "seedfree":
    pthrange = param.randTthresholdrange



for pth in pthrange:

    if param.seedtype == "seedbased":
        param.sig_threshold = pth
    elif param.seedtype == "seedfree":
        param.randTthreshold = pth

    msg = ""
    logging.info(msg)
    msg = "=========== signal threshold P = " + str(pth) + "% ==========="
    logging.info(msg)

    # -------------------------------------------
    #            Setup input directory
    # -------------------------------------------

    if param.seedtype == "seedbased":
        filein.indir = args.outdir + param.gsr + "_" + param.seedIDname + \
            "/" + param.sig_thresholdtype + str(param.sig_threshold) + ".0/"
    elif param.seedtype == "seedfree":
        filein.indir = args.outdir + param.gsr + "_" + param.seedIDname + \
            "/P" + str(param.randTthreshold) + ".0/"

    # --------------------------------------------------------
    # - Load QC info
    # --------------------------------------------------------

    df = pd.read_csv(filein.indir + 'QC_output.csv')
    msg = "Load QC data in: " + filein.indir + 'QC_output.csv'
    logging.info(msg)
    half = df["half"]
    n_cap = df["n_cap"]
    permlist = df["nperm"]
    del df

    # -----------------------------------------------------------
    #     Collect subjects distribution within two k solutions
    # -----------------------------------------------------------

    for k_solution in param.basis_k_range:

        clusterkidx_split1 = np.where((half == "training") & (n_cap == k_solution))[0]
        clusterpermlist_split1 = permlist[clusterkidx_split1].values

        clusterkidx_split2 = np.where((half == "test") & (n_cap == k_solution))[0]
        clusterpermlist_split2 = permlist[clusterkidx_split2].values
        
        msg = "Split 1 (k= " + str(k_solution) + ") permutation list (n=" + str(len(clusterpermlist_split1)) + "): " + \
            str(clusterpermlist_split1)
        logging.info(msg)        

        msg = "Split 2 (k= " + str(k_solution) + ") permutation list: (n=" + str(len(clusterpermlist_split2)) + "): " + \
            str(clusterpermlist_split2)
        logging.info(msg)

        # -----------------------------------------------------------
        #     Collect subjects distribution within this k solution
        # -----------------------------------------------------------

        permsub_split1 = getsplitlist_perm(
            permlist=clusterpermlist_split1, param=param, filein=filein, halftype="training")

        permsub_split2 = getsplitlist_perm(
            permlist=clusterpermlist_split2, param=param, filein=filein, halftype="test")

        msg = "Collected the distribution of subjects from permutations with k=" + str(k_solution) + " solution ..."
        logging.info(msg)            

        # -----------------------------------------------------------
        #     Count subjects within this k solution
        # -----------------------------------------------------------

        counts_split1 = pd.Series(permsub_split1).value_counts()
        counts_split2 = pd.Series(permsub_split2).value_counts()
        
        msg = "Counted the number of subjects from permutations with k=" + str(k_solution) + " solution ..."
        logging.info(msg)   

        # -----------------------------------------------------------
        #     Pool subject counts into the entire subject list
        # -----------------------------------------------------------

        counts_split1_f = np.empty((0, 1))
        counts_split2_f = np.empty((0, 1))

        for i in filein.sublist:
            tmp = np.array([[str(counts_split1.get(i))]])
            counts_split1_f = np.append(counts_split1_f, tmp, axis=0)
            del tmp

            tmp = np.array([[str(counts_split2.get(i))]])
            counts_split2_f = np.append(counts_split2_f, tmp, axis=0)
            del tmp

        counts_split1_f[counts_split1_f == 'None'] = '0'
        counts_split1_f = [int(i) for i in counts_split1_f]
        counts_split1_f = np.reshape(counts_split1_f, -1)

        counts_split2_f[counts_split2_f == 'None'] = '0'
        counts_split2_f = [int(i) for i in counts_split2_f]
        counts_split2_f = np.reshape(counts_split2_f, -1)

        permksolution = [k_solution] * (len(filein.sublist))
        permksolution = np.array(permksolution)

        msg = "Pooled the subject counts in to the entire subject list ..."
        logging.info(msg)

        # -----------------------------------------------------------
        #     Concatenate df (counts) from all cluster k
        # -----------------------------------------------------------
        if k_solution == param.basis_k_range[0]:
            df = pd.DataFrame({'subject': np.arange(1, len(filein.sublist)+1), 'subID': filein.sublist,
                               'k'+str(k_solution)+'_counts_split1': counts_split1_f,
                              'k'+str(k_solution)+'_counts_split2': counts_split2_f})
        else:
            df['k'+str(k_solution)+'_counts_split1'] = counts_split1_f
            df['k'+str(k_solution)+'_counts_split2'] = counts_split2_f

    # -----------------------------------------------------------
    #              Conmpute delta freq. in all k_solutions
    # -----------------------------------------------------------

    df['k5-k4_counts_split1'] = df['k5_counts_split1'] - df['k4_counts_split1']
    df['k5-k4_counts_split2'] = df['k5_counts_split2'] - df['k4_counts_split2']
    df['k5-k4_counts_mean12'] = ( df['k5-k4_counts_split1'] + df['k5-k4_counts_split2'] )/2
    df['k5-k4_counts_mean12_z'] = sp.stats.zscore(df['k5-k4_counts_mean12'])   

    # -----------------------------------------------------------
    #              Save results from all k_solution
    # -----------------------------------------------------------

    fig = plt.figure()
    ax = fig.add_subplot(111)
    sns.scatterplot(data=df, x='k4_counts_split1', y='k5_counts_split1',s=50, color="k")
    plt.title('Split 1 - k4 vs k5 counts in n=337 subjects')
    plt.xlim(120, 350)
    plt.ylim(120, 350)
    ax.set_aspect('equal', adjustable='box')
    savefilen = filein.outdir + 'P' + str(pth) + ".0_k4_subcount_split1_scatter.png"
    plt.savefig(savefilen, bbox_inches='tight')
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)    
    sns.scatterplot(data=df, x='k4_counts_split2', y='k5_counts_split2',s=50, color="k")
    plt.title('Split 2 - k4 vs k5 counts in n=337 subjects')
    plt.xlim(120, 350)
    plt.ylim(120, 350)
    ax.set_aspect('equal', adjustable='box')
    savefilen = filein.outdir + 'P' + str(pth) + ".0_k4_subcount_split2_scatter.png"
    plt.savefig(savefilen, bbox_inches='tight')
    plt.close()  

    fig = plt.figure()
    ax = fig.add_subplot(111)    
    sns.regplot(x=df["k5-k4_counts_split1"], y=df["k5-k4_counts_split2"], color="k", line_kws={"color":"r","alpha":0.7,"lw":3})
    r, p = sp.stats.pearsonr(x=df['k5-k4_counts_split1'], y=df['k5-k4_counts_split2'])
    plt.title('Split 1 vs 2, n=337 subjects (r=' + str(r) + ', p=' + str(p) + ')')
    plt.xlim(-150, 100)
    plt.ylim(-150, 100)
    ax.set_aspect('equal', adjustable='box')
    savefilen = filein.outdir + 'P' + str(pth) + ".0_k5-4_subcount_split12_scatter.png"
    plt.savefig(savefilen, bbox_inches='tight')
    plt.close()    
    
    fig = plt.figure()
    # ax = fig.add_subplot(111) 
    sns.histplot(data=df, x="k5-k4_counts_mean12_z", bins=40, kde=True, line_kws={"color":"g","alpha":0.7,"lw":3}, color='k')
    plt.rcParams['patch.linewidth'] = 0
    plt.title('Z-score: mean(sp1,2) of freq.(k5) - freq.(k4)')
    plt.xlim(-4, 4)
    # ax.set_aspect('equal', adjustable='box')
    savefilen = filein.outdir + 'P' + str(pth) + ".0_k5-4_subcount_meansplit12_Zdist.png"
    plt.savefig(savefilen, bbox_inches='tight')
    plt.close()   




    # -----------------------------------------------------------
    #            Identify 3 subgroups of subjects
    # -----------------------------------------------------------    
    for cTh in param.classTh:
    
        conditions = [df['k5-k4_counts_mean12_z'] < -cTh, df['k5-k4_counts_mean12_z'] > cTh]
        choices = [1,3]
        df['subgroup']=np.select(conditions, choices, default=2) 
        
        df_s1 = df[df['subgroup'] == 1]
        df_s1 = df_s1['subID']
        df_s2 = df[df['subgroup'] == 2]
        df_s2 = df_s2['subID']
        df_s3 = df[df['subgroup'] == 3]
        df_s3 = df_s3['subID']
    
        df_s1.to_csv(filein.outdir + 'subgroup1_' + str(cTh) + 'SD.csv',index=False)
        df_s2.to_csv(filein.outdir + 'subgroup2_' + str(cTh) + 'SD.csv',index=False)
        df_s3.to_csv(filein.outdir + 'subgroup3_' + str(cTh) + 'SD.csv',index=False)
    
    
        fig = plt.figure()
        ax = fig.add_subplot(111)   
        mypal=sns.color_palette("viridis_r", as_cmap=True)
        sns.scatterplot(data=df, x='k5-k4_counts_split1', y='k5-k4_counts_split2',hue='subgroup', palette=mypal,  s=50, edgecolor="black")
        r, p = sp.stats.pearsonr(x=df['k5-k4_counts_split1'], y=df['k5-k4_counts_split2'])
        plt.title('Split 1 vs 2, n=337 subjects (r=' + str(r) + ', p=' + str(p) + ')')
        plt.xlim(-150, 100)
        plt.ylim(-150, 100)
        ax.set_aspect('equal', adjustable='box')
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        savefilen = filein.outdir + 'P' + str(pth) + ".0_k5-4_subcount_split12_scatter_subgroup_" + str(cTh) + "SD.png"
        plt.savefig(savefilen, bbox_inches='tight')
        plt.close()    
    
        fig = plt.figure()
        ax = fig.add_subplot(111)     
        ax = df.hist(column='subgroup', bins=3,color='k',grid=False, rwidth=0.5)
        plt.title('# subjects in each subgroup')
        savefilen = filein.outdir + 'P' + str(pth) + ".0_k5-4_subcount_split12_numsubj_subgroup_" + str(cTh) + "SD.png"
        plt.savefig(savefilen, bbox_inches='tight')
        plt.close()     

        df.rename(columns={'subgroup':'subgroup_' + str(cTh) + "SD"}, inplace=True)
        
    df.to_csv(filein.outdir + 'P' + str(pth) + '.0_k_solution_subcounts_allcTh.csv')
    
    

# - Notify job completion
msg = "\n========== The jobs are all completed. ==========\n"
logging.info(msg)
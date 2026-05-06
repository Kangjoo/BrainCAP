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
import logging
import ptitprince as pt
import matplotlib.collections as clt
import matplotlib.pyplot as plt
from pycap_functions.pycap_postcap_loaddata import *
from pycap_functions.pycap_createWBimages import *
from pycap_functions.pycap_cap_splitcorrelation import *
from pycap_functions.pycap_cap_timemetrics import *
import seaborn as sns
from pandas.api.types import CategoricalDtype
from joypy import joyplot
from scipy import stats


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
                    filename=filein.logdir + 'pycap_Splithalf_postcap_timeanalytics_data.log',
                    filemode='w')
console = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)


# -------------------------------------------
#          Define classes.
# -------------------------------------------


class bcm_dur_tt:
    pass


bcm_dur_tt = bcm_dur_tt()


class bcv_dur_tt:
    pass


bcv_dur_tt = bcv_dur_tt()


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

    
    for stdk in param.basis_k_range:
        

        for sp in [1, 2]:
            if sp == 1:
                spdatatag = "training_data"
                sptitletag = "split_1"
            elif sp == 2:
                spdatatag = "test_data"
                sptitletag = "split_2"
            
            msg = "     <<  " + sptitletag + "  >>"
            logging.info(msg)
    
            
            # -------------------------------------------
            # - Load information about the new order of CAPs in the sorted CAP matrix
            # -------------------------------------------
            
            neworderpath1 = filein.outdir + "sorted_testcaps_neworder_" + sptitletag + "_allperms_basis" + str(stdk) + "_testk4.csv"
            neworderpath2 = filein.outdir + "sorted_testcaps_neworder_" + sptitletag + "_allperms_basis" + str(stdk) + "_testk5.csv"
            df1 = pd.read_csv(neworderpath1,index_col=0) 
            df2 = pd.read_csv(neworderpath2,index_col=0)
            neworder_allperm=pd.concat([df1, df2], axis=0, ignore_index=True)
            neworder_allperm = neworder_allperm.sort_values(by=['perm'], ascending=True, ignore_index=True)
            neworder_allperm_filen = filein.outdir + 'P' + str(pth) + "_" + sptitletag + '_SortedCAP_neworder_allperm.csv'
            neworder_allperm.to_csv(neworder_allperm_filen) 
            msg = "Collect and save the new orders of sorted CAPs in all permutations in " + neworder_allperm_filen
            logging.info(msg)
            del df1, df2, neworderpath1, neworderpath2
        
    
            # -----------------------------------
            # - For each permutation 
            # ----------------------------------- 
            
            for index in range(len(neworder_allperm)):

                
                # -----------------------------------
                # - Get the new order of CAPs to match with 5-basis CAPs
                # -----------------------------------
                perm = neworder_allperm["perm"][index]
                neworder = neworder_allperm['neworder'][index]
                neworder = neworder.replace(" ", "," )
                neworder = neworder[1:-1]
                neworder = list(map(int, neworder.split(",")))

                msg = "\n"
                logging.info(msg)
                msg = "     <<  perm " + str(perm) + ": " + str(len(neworder)) + " CAPs with new order " + str(neworder) + "  >>"
                logging.info(msg)
                msg = "\n"
                logging.info(msg)
                
                # -----------------------------------
                # - Load CAP maps (by default, it unfortunately loads both training/test datasets)
                # -----------------------------------    

                if param.seedtype == "seedbased":
                    filein.datapath = param.datadirtag + "/" + param.splittype + \
                        str(perm) + "/" + param.gsr + "_" + param.seedIDname + "/" + \
                        param.sig_thresholdtype + str(param.sig_threshold) + ".0/"
                elif param.seedtype == "seedfree":
                    filein.datapath = param.datadirtag + "/" + param.splittype + \
                        str(perm) + "/" + param.gsr + "_" + param.seedIDname + \
                        "/P" + str(param.randTthreshold) + ".0/"
                training_data, test_data = load_capoutput(filein=filein, param=param)


                # -----------------------------------
                # - Load split subject ID info
                # -----------------------------------
                msg = "============================================"
                logging.info(msg)
                msg = "Start loading sublist data from n=" + str(len(filein.sublist))
                logging.info(msg)
                
                splitdata_filen = filein.datapath + "subsplit_datalist.hdf5"
                f = h5py.File(splitdata_filen, 'r')
                training_sublist_idx = f['training_sublist_idx']
                test_sublist_idx = f['test_sublist_idx']

                training_data.splitlist = []
                for loaddata_perm in training_sublist_idx:
                    training_data.splitlist.append(filein.sublist[loaddata_perm])
                test_data.splitlist = []
                for loaddata_perm in test_sublist_idx:
                    test_data.splitlist.append(filein.sublist[loaddata_perm])        
                del f, training_sublist_idx, test_sublist_idx, loaddata_perm

                msg = "Load the sublist data in " + splitdata_filen
                logging.info(msg)
                msg = "    #(training list) = " + str(len(training_data.splitlist)) + \
                    ", #(test list) = " + str(len(test_data.splitlist))
                logging.info(msg)
                msg = "Training data list : " + str(np.unique(training_data.splitlist))
                logging.info(msg)
                msg = "Test data list : " + str(np.unique(test_data.splitlist))
                logging.info(msg)


                # -----------------------------------
                # - Select Split-half data
                # -----------------------------------
                if sp == 1:
                    inputdata = training_data
                elif sp == 2:
                    inputdata = test_data
                del training_data, test_data
                
                msg = "============================================\n"
                logging.info(msg)
                msg = ">> Selected CAP maps in " + sptitletag + " perm " + str(perm) + " " + str(inputdata.clmean.shape)
                logging.info(msg)    
  
                # - Compute CAP time metrics
                msg = "============================================\n"
                logging.info(msg)
                msg = ">> Compute CAP metrics for " + sptitletag + ".."
                logging.info(msg)
    
                # Between-CAP variance of fractional occupancy - group level
                grp_capdur, bcm_dur, bcv_dur = cap_occupancy_btwcap(data=inputdata, param=param)
    
                # Within-CAP between-subject variance of fractional occupancy - individual level
                capdur_ind_allcap, wcbsm_dur, wcbsv_dur = cap_occupancy_withincap_btwsubj(data=inputdata, dataname=sptitletag, filein=filein, param=param)
                
                # Within-CAP within-subject mean/variance of dwell time - individual level
                capdwt_wsm_allcap, capdwt_wsv_allcap = cap_dwelltime_withincap_withinsubj(data=inputdata, dataname=sptitletag, filein=filein, param=param)

    
                # -----------------------------------
                # - Reorder CAPs in the group level outputs to match with k-basis CAPs
                # -----------------------------------  
                linorder = capdur_ind_allcap.columns.values.tolist()
                linorder = linorder[1:]
                grp_capdur_r = grp_capdur[neworder]
                grp_capdur_r = np.array(grp_capdur_r).reshape(1,len(grp_capdur_r))
                grp_capdur_r=pd.DataFrame(grp_capdur_r, columns=linorder)
                df_warg = {'seedID': [param.seedIDname], 'seedthr': [param.randTthreshold], 'half': [sptitletag], 'stdk': [stdk], 'permidx': [perm]}
                col_warg = ['seedID', 'seedthr', 'half', 'stdk', 'permidx']
                df_info = pd.DataFrame(df_warg, columns=col_warg)
                grp_capdur_r = pd.concat([grp_capdur_r, df_info], axis=1)
                
                msg = "[Group-level] Between-CAP variance of occupancy: Recordered the output of CAPs to match with k-basis CAPs. \n" + str(grp_capdur_r)
                logging.info(msg)

                # -----------------------------------
                # - Reorder CAPs in the individual level outputs to match with k-basis CAPs - wcbsv_dur
                # -----------------------------------                  
                wcbsv_dur_r = wcbsv_dur[neworder]
                wcbsv_dur_r = np.array(wcbsv_dur_r).reshape(1,len(wcbsv_dur_r))
                wcbsv_dur_r=pd.DataFrame(wcbsv_dur_r, columns=linorder)
                df_warg = {'seedID': [param.seedIDname], 'seedthr': [param.randTthreshold], 'half': [sptitletag], 'stdk': [stdk], 'permidx': [perm]}
                col_warg = ['seedID', 'seedthr', 'half', 'stdk', 'permidx']
                df_info = pd.DataFrame(df_warg, columns=col_warg)
                wcbsv_dur_r = pd.concat([wcbsv_dur_r, df_info], axis=1)
                
                msg = "[Individual-level] Between-subject variance of occupancy in each CAP: Recordered the output of CAPs to match with k-basis CAPs. \n" + str(wcbsv_dur_r)
                logging.info(msg)

                # -----------------------------------
                # - Reorder CAPs in the individual level outputs to match with k-basis CAPs - wcbsm_dur
                # -----------------------------------                  
                wcbsm_dur_r = wcbsm_dur[neworder]
                wcbsm_dur_r = np.array(wcbsm_dur_r).reshape(1,len(wcbsm_dur_r))
                wcbsm_dur_r=pd.DataFrame(wcbsm_dur_r, columns=linorder)
                df_warg = {'seedID': [param.seedIDname], 'seedthr': [param.randTthreshold], 'half': [sptitletag], 'stdk': [stdk], 'permidx': [perm]}
                col_warg = ['seedID', 'seedthr', 'half', 'stdk', 'permidx']
                df_info = pd.DataFrame(df_warg, columns=col_warg)
                wcbsm_dur_r = pd.concat([wcbsm_dur_r, df_info], axis=1)
                
                msg = "[Individual-level] Mean of occupancy in each CAP: Recordered the output of CAPs to match with k-basis CAPs. \n" + str(wcbsm_dur_r)
                logging.info(msg) 
                
                # -----------------------------------
                # - Reorder CAPs in the individual level outputs to match with k-basis CAPs - capdur_ind_allcap
                # -----------------------------------  
                
                neworder = [str(x) for x in neworder]
                newcols = ['subID'] + neworder
                capdur_ind_allcap_r = capdur_ind_allcap[newcols]
                msg = "[Individual-level] Distribution of individual occupancy in each CAP: Recordered the output of CAPs to match with k-basis CAPs." 
                logging.info(msg)
                # Rename the estimated CAP columns
                capdur_ind_allcap_r.columns=capdur_ind_allcap.columns.values.tolist()
                
                # - Generate dataframe info columns
                permstdk = [stdk] * len(capdur_ind_allcap_r)
                permstdk = np.array(permstdk)
                permhalf = [sptitletag] * len(capdur_ind_allcap_r)
                permhalf = np.array(permhalf)
                permseedID = [param.seedIDname] * len(capdur_ind_allcap_r)
                permseedID = np.array(permseedID)
                permpermidx = [perm] * len(capdur_ind_allcap_r)
                permpermidx = np.array(permpermidx)
                if param.seedtype == "seedbased":
                    permthresh = [str(int(param.sig_threshold))] * len(capdur_ind_allcap_r)
                elif param.seedtype == "seedfree":
                    permthresh = [str(int(param.randTthreshold))] * len(capdur_ind_allcap_r)
                permthresh = np.array(permthresh)
                df_warg = {'seedID': permseedID, 'seedthr': permthresh, 'half': permhalf, 'stdk': permstdk, 'permidx': permpermidx}
                col_warg = ['seedID', 'seedthr', 'half', 'stdk', 'permidx']
                df_info = pd.DataFrame(df_warg, columns=col_warg)
                # - Add to DataFrame    
                capdur_ind_allcap_r = pd.concat([capdur_ind_allcap_r, df_info], axis=1)
                
                msg = "\n" + str(capdur_ind_allcap_r) 
                logging.info(msg)    
                
                
                # -----------------------------------
                # - Reorder CAPs in the individual level outputs to match with k-basis CAPs - capdwt_wsm_allcap
                # -----------------------------------  
                
                neworder = [str(x) for x in neworder]
                newcols = ['subID'] + neworder
                capdwt_wsm_allcap_r = capdwt_wsm_allcap[newcols]
                msg = "[Individual-level] Distribution of individual within-subject mean of dwell time in each CAP: Recordered the output of CAPs to match with k-basis CAPs." 
                logging.info(msg)
                # Rename the estimated CAP columns
                capdwt_wsm_allcap_r.columns=capdwt_wsm_allcap.columns.values.tolist()
                
                # - Generate dataframe info columns
                permstdk = [stdk] * len(capdwt_wsm_allcap_r)
                permstdk = np.array(permstdk)
                permhalf = [sptitletag] * len(capdwt_wsm_allcap_r)
                permhalf = np.array(permhalf)
                permseedID = [param.seedIDname] * len(capdwt_wsm_allcap_r)
                permseedID = np.array(permseedID)
                permpermidx = [perm] * len(capdwt_wsm_allcap_r)
                permpermidx = np.array(permpermidx)
                if param.seedtype == "seedbased":
                    permthresh = [str(int(param.sig_threshold))] * len(capdwt_wsm_allcap_r)
                elif param.seedtype == "seedfree":
                    permthresh = [str(int(param.randTthreshold))] * len(capdwt_wsm_allcap_r)
                permthresh = np.array(permthresh)
                df_warg = {'seedID': permseedID, 'seedthr': permthresh, 'half': permhalf, 'stdk': permstdk, 'permidx': permpermidx}
                col_warg = ['seedID', 'seedthr', 'half', 'stdk', 'permidx']
                df_info = pd.DataFrame(df_warg, columns=col_warg)
                # - Add to DataFrame    
                capdwt_wsm_allcap_r = pd.concat([capdwt_wsm_allcap_r, df_info], axis=1)
                
                msg = "\n" + str(capdwt_wsm_allcap_r) 
                logging.info(msg)                 
                
                
                
                # -----------------------------------
                # - Reorder CAPs in the individual level outputs to match with k-basis CAPs - capdwt_wsv_allcap
                # -----------------------------------  
                
                neworder = [str(x) for x in neworder]
                newcols = ['subID'] + neworder
                capdwt_wsv_allcap_r = capdwt_wsv_allcap[newcols]
                msg = "[Individual-level] Distribution of individual within-subject variance of dwell time in each CAP: Recordered the output of CAPs to match with k-basis CAPs." 
                logging.info(msg)
                # Rename the estimated CAP columns
                capdwt_wsv_allcap_r.columns=capdwt_wsv_allcap.columns.values.tolist()
                
                # - Generate dataframe info columns
                permstdk = [stdk] * len(capdwt_wsv_allcap_r)
                permstdk = np.array(permstdk)
                permhalf = [sptitletag] * len(capdwt_wsv_allcap_r)
                permhalf = np.array(permhalf)
                permseedID = [param.seedIDname] * len(capdwt_wsv_allcap_r)
                permseedID = np.array(permseedID)
                permpermidx = [perm] * len(capdwt_wsv_allcap_r)
                permpermidx = np.array(permpermidx)
                if param.seedtype == "seedbased":
                    permthresh = [str(int(param.sig_threshold))] * len(capdwt_wsv_allcap_r)
                elif param.seedtype == "seedfree":
                    permthresh = [str(int(param.randTthreshold))] * len(capdwt_wsv_allcap_r)
                permthresh = np.array(permthresh)
                df_warg = {'seedID': permseedID, 'seedthr': permthresh, 'half': permhalf, 'stdk': permstdk, 'permidx': permpermidx}
                col_warg = ['seedID', 'seedthr', 'half', 'stdk', 'permidx']
                df_info = pd.DataFrame(df_warg, columns=col_warg)
                # - Add to DataFrame    
                capdwt_wsv_allcap_r = pd.concat([capdwt_wsv_allcap_r, df_info], axis=1)
                
                msg = "\n" + str(capdwt_wsv_allcap_r) 
                logging.info(msg)                  
                
                
                
                
                del df_info, permstdk, permhalf, permseedID, permthresh, df_warg, col_warg, inputdata
            
                
                
                # -----------------------------------
                #- Combine df over all permutations
                # -----------------------------------  

                if 'grp_capdur_r_allperm' in locals():
                    grp_capdur_r_allperm = pd.concat([grp_capdur_r_allperm, grp_capdur_r], axis=0, ignore_index=True)
                else:
                    grp_capdur_r_allperm = grp_capdur_r
                    
                if 'wcbsv_dur_r_allperm' in locals():
                    wcbsv_dur_r_allperm = pd.concat([wcbsv_dur_r_allperm, wcbsv_dur_r], axis=0, ignore_index=True)
                else:
                    wcbsv_dur_r_allperm = wcbsv_dur_r

                if 'wcbsm_dur_r_allperm' in locals():
                    wcbsm_dur_r_allperm = pd.concat([wcbsm_dur_r_allperm, wcbsm_dur_r], axis=0, ignore_index=True)
                else:
                    wcbsm_dur_r_allperm = wcbsm_dur_r                    
                    
                if 'capdur_ind_allcap_r_allperm' in locals():
                    capdur_ind_allcap_r_allperm = pd.concat([capdur_ind_allcap_r_allperm, capdur_ind_allcap_r], axis=0, ignore_index=True)
                else:
                    capdur_ind_allcap_r_allperm = capdur_ind_allcap_r    
                    
                if 'capdwt_wsm_allcap_r_allperm' in locals():
                    capdwt_wsm_allcap_r_allperm = pd.concat([capdwt_wsm_allcap_r_allperm, capdwt_wsm_allcap_r], axis=0, ignore_index=True)
                else:
                    capdwt_wsm_allcap_r_allperm = capdwt_wsm_allcap_r  
                    
                if 'capdwt_wsv_allcap_r_allperm' in locals():
                    capdwt_wsv_allcap_r_allperm = pd.concat([capdwt_wsv_allcap_r_allperm, capdwt_wsv_allcap_r], axis=0, ignore_index=True)
                else:
                    capdwt_wsv_allcap_r_allperm = capdwt_wsv_allcap_r                        
                    
                    
                msg = "grp_capdur_r_allperm " + str(grp_capdur_r_allperm.shape)
                logging.info(msg)
                msg = "wcbsv_dur_r_allperm " + str(wcbsv_dur_r_allperm.shape)
                logging.info(msg)
                msg = "wcbsm_dur_r_allperm " + str(wcbsm_dur_r_allperm.shape)
                logging.info(msg)                
                msg = "capdur_ind_allcap_r_allperm " + str(capdur_ind_allcap_r_allperm.shape)
                logging.info(msg)
                msg = "capdwt_wsm_allcap_r_allperm " + str(capdwt_wsm_allcap_r_allperm.shape)
                logging.info(msg)                   
                msg = "capdwt_wsv_allcap_r_allperm " + str(capdwt_wsv_allcap_r_allperm.shape)
                logging.info(msg)                
                
                del grp_capdur_r, capdur_ind_allcap_r
            
    # Fill zero occupancy for 5th CAP 
    grp_capdur_r_allperm['4'] = grp_capdur_r_allperm['4'].fillna(0)
    wcbsv_dur_r_allperm['4'] = wcbsv_dur_r_allperm['4'].fillna(0)
    wcbsm_dur_r_allperm['4'] = wcbsm_dur_r_allperm['4'].fillna(0)
    capdur_ind_allcap_r_allperm['4'] = capdur_ind_allcap_r_allperm['4'].fillna(0)
    capdwt_wsm_allcap_r_allperm['4'] = capdwt_wsm_allcap_r_allperm['4'].fillna(0)
    capdwt_wsv_allcap_r_allperm['4'] = capdwt_wsv_allcap_r_allperm['4'].fillna(0)
       
    # Save results
    grp_capdur_filen = filein.outdir + 'P' + str(pth) + '_Occupancy_group.csv'
    grp_capdur_r_allperm.to_csv(grp_capdur_filen)   
    wcbsv_dur_filen = filein.outdir + 'P' + str(pth) + '_Occupancy_wcbsv.csv'
    wcbsv_dur_r_allperm.to_csv(wcbsv_dur_filen)
    wcbsm_dur_filen = filein.outdir + 'P' + str(pth) + '_Occupancy_wcbsm.csv'
    wcbsm_dur_r_allperm.to_csv(wcbsm_dur_filen)
    capdur_ind_filen = filein.outdir + 'P' + str(pth) + '_Occupancy_individual.csv'
    capdur_ind_allcap_r_allperm.to_csv(capdur_ind_filen)
    capdwt_wsm_filen = filein.outdir + 'P' + str(pth) + '_DwellTime_individual_WSmean.csv'
    capdwt_wsm_allcap_r_allperm.to_csv(capdwt_wsm_filen)      
    capdwt_wsv_filen = filein.outdir + 'P' + str(pth) + '_DwellTime_individual_WSstd.csv'
    capdwt_wsv_allcap_r_allperm.to_csv(capdwt_wsv_filen)     
    
    
    msg="Saving the results from temporal CAP metrics in \n" + grp_capdur_filen + "\n" + wcbsv_dur_filen + "\n" + wcbsm_dur_filen + "\n" + capdur_ind_filen + "\n" + capdwt_wsm_filen + "\n" + capdwt_wsv_filen
    logging.info(msg)
    
        
    
       



# - Notify job completion
msg = "\n========== The jobs are all completed. ==========\n"
logging.info(msg)

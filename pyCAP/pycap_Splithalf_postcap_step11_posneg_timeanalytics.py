#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Kangjoo Lee (kangjoo.lee@yale.edu)
# Created Date: 10/26/2022
# Last Updated: 10/26/2022
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
import nibabel as nib
import numpy as np
import argparse
import itertools
import pandas as pd
import scipy as sp
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
import copy
from matplotlib.colors import to_rgb
from matplotlib.collections import PolyCollection
from mpl_toolkits import mplot3d
import matplotlib.colors as mpc


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
if param.seedtype == "seedfree":
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
                    filename=filein.logdir + 'output_postcap_step11.log',
                    filemode='w')
console = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)



# -------------------------------------------
#          Define functions.
# -------------------------------------------

def convert_capdata_indscore(capdata,filein,capn):
    capdata_ind=np.zeros((len(filein.sublist),),dtype=float)
    i=0
    for subid in filein.sublist:
        subdata = capdata.loc[(capdata['subID'] == subid)]
        capdata_ind[i,]=subdata[capn].mean()
        i=i+1
    return capdata_ind



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

if param.seedtype == "seedfree":
    pthrange = param.randTthresholdrange



                
# -------------------------------------------
#          Visualization starts here.
# -------------------------------------------



msg="\n\n"
logging.info(msg)
msg="Positive vs Negative CAP comparisions of temporal metrics."
logging.info(msg)
msg="\n\n"
logging.info(msg)




for pth in pthrange:

    if param.seedtype == "seedfree":
        param.randTthreshold = pth
        
        
    # Define variables
    colcol = ['0','1','2','3','4'] 
    mypal=sns.color_palette("viridis_r", as_cmap=True)
    mypal2=[[0,0,0],[0,0,0],[0,0,0]]    
    mypal3=[[0.5,0.5,0.5],[1,1,1]]
    colors = sns.color_palette()

    # # -------------------------------------------------------------
    # # 
    # #                       LOAD DATA & QC
    # # 
    # # -------------------------------------------------------------

    # Load data

    capdur_ind_filen = filein.outdir + 'P' + str(pth) + '_Occupancy_individual_trt_subgroup.csv'
    capdur_ind_allcap_r_allday_allperm = pd.read_csv(capdur_ind_filen, index_col=0)
    
    capdwt_wsm_filen = filein.outdir + 'P' + str(pth) + '_DwellTime_individual_WSmean_trt_subgroup.csv'
    capdwt_wsm_allcap_r_allday_allperm = pd.read_csv(capdwt_wsm_filen, index_col=0)
    capdwt_wsv_filen = filein.outdir + 'P' + str(pth) + '_DwellTime_individual_WSstd_trt_subgroup.csv'
    capdwt_wsv_allcap_r_allday_allperm = pd.read_csv(capdwt_wsv_filen, index_col=0)


    
    # Individual level QC - capdur_ind_allcap_r_allday_allperm
    msg = "capdur_ind_allcap_r_allday_allperm " + str(capdur_ind_allcap_r_allday_allperm.shape)
    logging.info(msg)
    for day_n in [1, 2]:
            
        for stdk in param.basis_k_range:
            dataset = capdur_ind_allcap_r_allday_allperm[(capdur_ind_allcap_r_allday_allperm['day'] == day_n) & (capdur_ind_allcap_r_allday_allperm['stdk'] == stdk) ]
            msg = "[Day " + str(day_n) + "] capdur_ind_allcap_r_allday_allperm (k=" + str(stdk) +") =" + str(dataset.shape)
            logging.info(msg)        


    # Individual level QC - capdwt_wsm_allcap_r_allday_allperm
    msg = "capdwt_wsm_allcap_r_allday_allperm " + str(capdwt_wsm_allcap_r_allday_allperm.shape)
    logging.info(msg)
    for day_n in [1, 2]:
            
        for stdk in param.basis_k_range:
            dataset = capdwt_wsm_allcap_r_allday_allperm[(capdwt_wsm_allcap_r_allday_allperm['day'] == day_n) & (capdwt_wsm_allcap_r_allday_allperm['stdk'] == stdk) ]
            msg = "[Day " + str(day_n) + "] capdwt_wsm_allcap_r_allday_allperm (k=" + str(stdk) +") =" + str(dataset.shape)
            logging.info(msg)     


    # Individual level QC - capdwt_wsv_allcap_r_allday_allperm
    msg = "capdwt_wsv_allcap_r_allday_allperm " + str(capdwt_wsv_allcap_r_allday_allperm.shape)
    logging.info(msg)
    for day_n in [1, 2]:
            
        for stdk in param.basis_k_range:
            dataset = capdwt_wsv_allcap_r_allday_allperm[(capdwt_wsv_allcap_r_allday_allperm['day'] == day_n) & (capdwt_wsv_allcap_r_allday_allperm['stdk'] == stdk) ]
            msg = "[Day " + str(day_n) + "] capdwt_wsv_allcap_r_allday_allperm (k=" + str(stdk) +") =" + str(dataset.shape)
            logging.info(msg) 



    # # -------------------------------------------------------------
    # # 
    # #                           PLOT DATA 
    # # 
    # # -------------------------------------------------------------


        # ---------- plot CAP 0 vs 1 for DwellTime_WSmean ------------#
        dataname="DwellTime_WSmean"
        capdata_pos=capdwt_wsm_allcap_r_allday_allperm[["subID", "0", "half","permidx", "day"]]
        capdata_neg=capdwt_wsm_allcap_r_allday_allperm[["subID", "1", "half","permidx", "day"]]
        capdata_pos=convert_capdata_indscore(capdata_pos,filein,"0")
        capdata_neg=convert_capdata_indscore(capdata_neg,filein,"1")
        df = pd.DataFrame(columns=['capdata_pos', 'capdata_neg'])
        df["capdata_pos"]=capdata_pos
        df["capdata_neg"]=capdata_neg
        
        fig = plt.figure()
        ax = fig.add_subplot(111)    
        sns.regplot(x=df["capdata_pos"], y=df["capdata_neg"], color="k", line_kws={"color":"r","alpha":0.7,"lw":3})
        r, p = sp.stats.pearsonr(x=df['capdata_pos'], y=df['capdata_neg'])
        plt.title('CAP 0 vs 1 (r=' + str(r) + ', p=' + str(p) + ')')
        plt.xlim(1, 6)
        plt.ylim(1, 6)
        ax.set_aspect('equal', adjustable='box')
        savefilen = filein.outdir + 'P' + str(pth) + ".0_CAPposneg01_" + dataname + "_scatter.png"
        plt.savefig(savefilen, bbox_inches='tight')
        plt.close()      
        del dataname, capdata_pos, capdata_neg, df, fig, ax
        
        # ---------- plot CAP 2 vs 3 for DwellTime_WSmean ------------#
        dataname="DwellTime_WSmean"
        capdata_pos=capdwt_wsm_allcap_r_allday_allperm[["subID", "2", "half","permidx", "day"]]
        capdata_neg=capdwt_wsm_allcap_r_allday_allperm[["subID", "3", "half","permidx", "day"]]
        capdata_pos=convert_capdata_indscore(capdata_pos,filein,"2")
        capdata_neg=convert_capdata_indscore(capdata_neg,filein,"3")
        df = pd.DataFrame(columns=['capdata_pos', 'capdata_neg'])
        df["capdata_pos"]=capdata_pos
        df["capdata_neg"]=capdata_neg
        
        fig = plt.figure()
        ax = fig.add_subplot(111)    
        sns.regplot(x=df["capdata_pos"], y=df["capdata_neg"], color="k", line_kws={"color":"r","alpha":0.7,"lw":3})
        r, p = sp.stats.pearsonr(x=df['capdata_pos'], y=df['capdata_neg'])
        plt.title('CAP 2 vs 3 (r=' + str(r) + ', p=' + str(p) + ')')
        plt.xlim(1, 6)
        plt.ylim(1, 6)
        ax.set_aspect('equal', adjustable='box')
        savefilen = filein.outdir + 'P' + str(pth) + ".0_CAPposneg23_" + dataname + "_scatter.png"
        plt.savefig(savefilen, bbox_inches='tight')
        plt.close()      
        del dataname, capdata_pos, capdata_neg, df, fig, ax







        # ---------- plot CAP 0 vs 1 for DwellTime_WSstd ------------#
        dataname="DwellTime_WSstd"
        capdata_pos=capdwt_wsv_allcap_r_allday_allperm[["subID", "0", "half","permidx", "day"]]
        capdata_neg=capdwt_wsv_allcap_r_allday_allperm[["subID", "1", "half","permidx", "day"]]
        capdata_pos=convert_capdata_indscore(capdata_pos,filein,"0")
        capdata_neg=convert_capdata_indscore(capdata_neg,filein,"1")
        df = pd.DataFrame(columns=['capdata_pos', 'capdata_neg'])
        df["capdata_pos"]=capdata_pos
        df["capdata_neg"]=capdata_neg
        
        fig = plt.figure()
        ax = fig.add_subplot(111)    
        sns.regplot(x=df["capdata_pos"], y=df["capdata_neg"], color="k", line_kws={"color":"r","alpha":0.7,"lw":3})
        r, p = sp.stats.pearsonr(x=df['capdata_pos'], y=df['capdata_neg'])
        plt.title('CAP 0 vs 1 (r=' + str(r) + ', p=' + str(p) + ')')
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        ax.set_aspect('equal', adjustable='box')
        savefilen = filein.outdir + 'P' + str(pth) + ".0_CAPposneg01_" + dataname + "_scatter.png"
        plt.savefig(savefilen, bbox_inches='tight')
        plt.close()      
        del dataname, capdata_pos, capdata_neg, df, fig, ax
        
        # ---------- plot CAP 2 vs 3 for DwellTime_WSstd ------------#
        dataname="DwellTime_WSstd"
        capdata_pos=capdwt_wsv_allcap_r_allday_allperm[["subID", "2", "half","permidx", "day"]]
        capdata_neg=capdwt_wsv_allcap_r_allday_allperm[["subID", "3", "half","permidx", "day"]]
        capdata_pos=convert_capdata_indscore(capdata_pos,filein,"2")
        capdata_neg=convert_capdata_indscore(capdata_neg,filein,"3")
        df = pd.DataFrame(columns=['capdata_pos', 'capdata_neg'])
        df["capdata_pos"]=capdata_pos
        df["capdata_neg"]=capdata_neg
        
        fig = plt.figure()
        ax = fig.add_subplot(111)    
        sns.regplot(x=df["capdata_pos"], y=df["capdata_neg"], color="k", line_kws={"color":"r","alpha":0.7,"lw":3})
        r, p = sp.stats.pearsonr(x=df['capdata_pos'], y=df['capdata_neg'])
        plt.title('CAP 2 vs 3 (r=' + str(r) + ', p=' + str(p) + ')')
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        ax.set_aspect('equal', adjustable='box')
        savefilen = filein.outdir + 'P' + str(pth) + ".0_CAPposneg23_" + dataname + "_scatter.png"
        plt.savefig(savefilen, bbox_inches='tight')
        plt.close()      
        del dataname, capdata_pos, capdata_neg, df, fig, ax

 






        # ---------- plot CAP 0 vs 1 for Occupancy_individual ------------#
        dataname="Occupancy_individual"
        capdata_pos=capdur_ind_allcap_r_allday_allperm[["subID", "0", "half","permidx", "day"]]
        capdata_neg=capdur_ind_allcap_r_allday_allperm[["subID", "1", "half","permidx", "day"]]
        capdata_pos=convert_capdata_indscore(capdata_pos,filein,"0")
        capdata_neg=convert_capdata_indscore(capdata_neg,filein,"1")
        df = pd.DataFrame(columns=['capdata_pos', 'capdata_neg'])
        df["capdata_pos"]=capdata_pos
        df["capdata_neg"]=capdata_neg
        
        fig = plt.figure()
        ax = fig.add_subplot(111)    
        sns.regplot(x=df["capdata_pos"], y=df["capdata_neg"], color="k", line_kws={"color":"r","alpha":0.7,"lw":3})
        r, p = sp.stats.pearsonr(x=df['capdata_pos'], y=df['capdata_neg'])
        plt.title('CAP 0 vs 1 (r=' + str(r) + ', p=' + str(p) + ')')
        plt.xlim(0, 0.3)
        plt.ylim(0, 0.3)
        ax.set_aspect('equal', adjustable='box')
        savefilen = filein.outdir + 'P' + str(pth) + ".0_CAPposneg01_" + dataname + "_scatter.png"
        plt.savefig(savefilen, bbox_inches='tight')
        plt.close()      
        del dataname, capdata_pos, capdata_neg, df, fig, ax
        
        # ---------- plot CAP 2 vs 3 for Occupancy_individual ------------#
        dataname="Occupancy_individual"
        capdata_pos=capdur_ind_allcap_r_allday_allperm[["subID", "2", "half","permidx", "day"]]
        capdata_neg=capdur_ind_allcap_r_allday_allperm[["subID", "3", "half","permidx", "day"]]
        capdata_pos=convert_capdata_indscore(capdata_pos,filein,"2")
        capdata_neg=convert_capdata_indscore(capdata_neg,filein,"3")
        df = pd.DataFrame(columns=['capdata_pos', 'capdata_neg'])
        df["capdata_pos"]=capdata_pos
        df["capdata_neg"]=capdata_neg
        
        fig = plt.figure()
        ax = fig.add_subplot(111)    
        sns.regplot(x=df["capdata_pos"], y=df["capdata_neg"], color="k", line_kws={"color":"r","alpha":0.7,"lw":3})
        r, p = sp.stats.pearsonr(x=df['capdata_pos'], y=df['capdata_neg'])
        plt.title('CAP 2 vs 3 (r=' + str(r) + ', p=' + str(p) + ')')
        plt.xlim(0, 0.3)
        plt.ylim(0, 0.3)
        ax.set_aspect('equal', adjustable='box')
        savefilen = filein.outdir + 'P' + str(pth) + ".0_CAPposneg23_" + dataname + "_scatter.png"
        plt.savefig(savefilen, bbox_inches='tight')
        plt.close()      
        del dataname, capdata_pos, capdata_neg, df, fig, ax




# - Notify job completion
msg = "\n========== The jobs are all completed. ==========\n"
logging.info(msg)











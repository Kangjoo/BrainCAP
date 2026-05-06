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
import copy
from matplotlib.colors import to_rgb
from matplotlib.collections import PolyCollection
from mpl_toolkits import mplot3d
import matplotlib.colors as mpc
plt.switch_backend('agg')


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
parser.add_argument("-sg", "--subgroupfilen", dest="subgroupfile", required=True,
                    help="Subgroup filename", type=lambda f: open(f))
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
filein.subgroup_filen = args.subgroupfile.name

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
                    filename=filein.logdir + 'pycap_Splithalf_postcap_indpermrel_timeanalytics.log',
                    filemode='w')
console = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)



# -------------------------------------------
#          Define functions.
# -------------------------------------------


def f_test(x, y):
    x = np.array(x)
    y = np.array(y)
    f = np.var(x, ddof=1)/np.var(y, ddof=1) #calculate F test statistic 
    dfn = x.size-1 #define degrees of freedom numerator 
    dfd = y.size-1 #define degrees of freedom denominator 
    p = 1-stats.f.cdf(f, dfn, dfd) #find p-value of F test statistic 
    return f, p


def plot_heatmap(capheatmap_sorted,capsubgrpbar_sorted,filein,sp,day,capn,capheatlmap_filen,cmin,cmax):
    fig, (ax1, ax2) = plt.subplots(2,1)
    capmap=ax1.imshow(capheatmap_sorted, cmap='brg', interpolation='nearest')
    fig.colorbar(capmap, ax=ax1)
    capmap.set_clim(cmin,cmax)
    cmap = plt.get_cmap('viridis_r', np.max(capsubgrpbar_sorted) - np.min(capsubgrpbar_sorted) + 1)         
    grpbar=ax2.matshow(capsubgrpbar_sorted, cmap=cmap, vmin=np.min(capsubgrpbar_sorted) - 0.5, vmax=np.max(capsubgrpbar_sorted) + 0.5)
    fig.colorbar(grpbar, ticks=np.arange(np.min(capsubgrpbar_sorted), np.max(capsubgrpbar_sorted) + 1))
    
    plt.savefig(capheatlmap_filen)
    return 


def capmetric_ind_heatmap(capdata,filein,param,dataname,pth,capn,cmin,cmax):

    for sp in [1,2]:
        capdata_sp=capdata.loc[(capdata["half"] == "split_" + str(sp))]
       
        for day in [1,2]:
            capdata_sp_day = capdata_sp.loc[capdata_sp["day"] == day]
            
            capheatmap = np.empty([len(filein.sublist),param.maxperm], dtype=float)
            capheatmap.fill(np.nan) 
            capsubgrpbar = np.empty([len(filein.sublist),param.maxperm], dtype=float)
            capsubgrpbar.fill(np.nan)                
            
            
            # collect heatmat data matrices across permutations
            for permidx in np.arange(param.minperm,param.maxperm+1):
            
                permdata = capdata_sp_day.loc[capdata_sp_day["permidx"] == permidx] # 168 x 5 columns (subID, canp, permidx, day, subgroup_30nf)
                permdata = permdata[["subID",capn,"subgroup_30nf"]]
               
                i=0
                for subid in filein.sublist:
                    subdata = permdata.loc[(permdata['subID'] == subid)]
                    if subdata.empty:
                        msg="Perm " + str(permidx) + ": Subject " + str(subid) + " is not in this split"
                        logging.info(msg)
                    else:
                        msg="Perm " + str(permidx) + ": Subject " + str(subid) + " is added to heatmap data : " + str(subdata[capn])
                        logging.info(msg)
                        capheatmap[i,permidx-1] = subdata[capn]
                        capsubgrpbar[i,permidx-1] =subdata["subgroup_30nf"].astype(int)
                    i=i+1
            
            # rank-order subjects using the mean value over data collected across permutations
            if 'sortidx' in locals():
                meanval=np.nanmean(capheatmap, axis=1) #(337,)
                meanval_sgr=np.nanmean(capsubgrpbar, axis=1) #(337,)
                capheatmap_sorted=capheatmap[sortidx,]
                capsubgrpbar_sorted=meanval_sgr[sortidx,]     
                capsubgrpbar_sorted=capsubgrpbar_sorted.astype(int) 
                capsubgrpbar_sorted=np.expand_dims(capsubgrpbar_sorted, axis=1)
                capsubgrpbar_sorted=np.repeat(capsubgrpbar_sorted,param.maxperm,axis=1)  
            else:
                meanval=np.nanmean(capheatmap, axis=1) #(337,)
                meanval_sgr=np.nanmean(capsubgrpbar, axis=1) #(337,)
                sortidx=np.argsort(-meanval)
                capheatmap_sorted=capheatmap[sortidx,]
                capsubgrpbar_sorted=meanval_sgr[sortidx,]     
                capsubgrpbar_sorted=capsubgrpbar_sorted.astype(int) 
                capsubgrpbar_sorted=np.expand_dims(capsubgrpbar_sorted, axis=1)
                capsubgrpbar_sorted=np.repeat(capsubgrpbar_sorted,param.maxperm,axis=1)   
            
            # plot heatmap and save
            capheatlmap_filen =  filein.outdir + 'P' + str(pth) + '_' + dataname + '_split' + str(sp) + '_day' + str(day) + '_cap' + str(capn) + '.png'
            plot_heatmap(capheatmap_sorted,capsubgrpbar_sorted,filein,sp,day,capn,capheatlmap_filen,cmin,cmax)
            
            del capdata_sp_day, capheatmap, capsubgrpbar,capheatmap_sorted, capsubgrpbar_sorted, i,meanval, meanval_sgr
        del capdata_sp
    return 




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



                
# -------------------------------------------
#          Visualization starts here.
# -------------------------------------------



msg="\n\n"
logging.info(msg)
msg="Individual level Reliability of time analytics starts."
logging.info(msg)
msg="\n\n"
logging.info(msg)




for pth in pthrange:

    if param.seedtype == "seedbased":
        param.sig_threshold = pth
    elif param.seedtype == "seedfree":
        param.randTthreshold = pth
        
        
    # Define variables
    colcol = ['0'] #['0','1','2','3','4'] 
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

    # for capn in colcol:
    #     capdata=capdur_ind_allcap_r_allday_allperm[["subID", capn, "half","permidx", "day", "subgroup_30nf"]]
    #     dataname="Occupancy_individual"
    #     capmetric_ind_heatmap(capdata,filein,param,dataname,pth,capn,0,0.3)
    # del capdata, dataname, capn 
   

    for capn in colcol:
        capdata=capdwt_wsm_allcap_r_allday_allperm[["subID", capn, "half","permidx", "day", "subgroup_30nf"]]
        dataname="DwellTime_WSmean"
        capmetric_ind_heatmap(capdata,filein,param,dataname,pth,capn,2,4.5)
    del capdata, dataname, capn      


    # for capn in colcol:
    #     capdata=capdwt_wsv_allcap_r_allday_allperm[["subID", capn, "half","permidx", "day", "subgroup_30nf"]]
    #     dataname="DwellTime_WSstd"
    #     capmetric_ind_heatmap(capdata,filein,param,dataname,pth,capn,0,8)
    # del capdata, dataname, capn            





# - Notify job completion
msg = "\n========== The jobs are all completed. ==========\n"
logging.info(msg)











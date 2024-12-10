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
import matplotlib.pyplot as plt
plt.rcParams['figure.max_open_warning'] = 50


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
                    filename=filein.logdir + 'pycap_Splithalf_postcap_TRT_timeanalytics_plot.log',
                    filemode='w')
console = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)



# -------------------------------------------
#          Define functions.
# -------------------------------------------

def select_daydata(inputdata, day_n):

    dayidx=np.where(inputdata.frame_day == day_n)[0]
    msg = ">> ---- Day " + str(day_n) + " index: " + str(len(dayidx))
    logging.info(msg)
    
    
    daydata = copy.deepcopy(inputdata)
    daydata.frame_clusterID = inputdata.frame_clusterID[dayidx,]
    daydata.frame_subID = inputdata.frame_subID[dayidx,]
    daydata.frame_day = inputdata.frame_day[dayidx,]
    
    msg = ">> ---- Day " + str(day_n) + " data: clmean " + str(daydata.clmean.shape) + \
          ", frame_clusterID " + str(daydata.frame_clusterID.shape) + \
          ", frame_subID " + str(daydata.frame_subID.shape) + \
          ", frame_day " + str(daydata.frame_day.shape)
    logging.info(msg)

    return daydata


def add_subgroup_data_ksolution(dataset, filein, cTh):

    cThstring = '_' + str(cTh) + 'SD'
    cThcol = ['subgroup' + cThstring]
    
    # Load subgroup defitnitions
    subgroup1 = pd.read_csv(filein.outdir + 'subgroup1' + cThstring + '.csv')
    subgroup2 = pd.read_csv(filein.outdir + 'subgroup2' + cThstring + '.csv')
    subgroup3 = pd.read_csv(filein.outdir + 'subgroup3' + cThstring + '.csv')

    # Assign subgroup defitnitions
    dataset.loc[:,cThcol]=np.nan
    sgidx1= dataset.index[dataset["subID"].isin(subgroup1["subID"])]
    dataset.loc[sgidx1,cThcol]=1
    sgidx2= dataset.index[dataset["subID"].isin(subgroup2["subID"])]
    dataset.loc[sgidx2,cThcol]=2
    sgidx3= dataset.index[dataset["subID"].isin(subgroup3["subID"])]
    dataset.loc[sgidx3,cThcol]=3
    del sgidx1, sgidx2,sgidx3
    
    return dataset


def add_subgroup_data_nfcluster(dataset, filein, cTh):
    
    cThstring = '_' + str(cTh) + 'nf'
    cThcol = ['subgroup' + cThstring]
    
    # Load subgroup defitnitions
    subgroup1 = pd.read_csv(filein.outdir + 'subgroup1' + cThstring + '.csv')
    subgroup2 = pd.read_csv(filein.outdir + 'subgroup2' + cThstring + '.csv')
    subgroup3 = pd.read_csv(filein.outdir + 'subgroup3' + cThstring + '.csv')
     
    # Assign subgroup defitnitions
    dataset.loc[:,cThcol]=np.nan
    sgidx1= dataset.index[dataset["subID"].isin(subgroup1["subID"])]
    dataset.loc[sgidx1,cThcol]=1
    sgidx2= dataset.index[dataset["subID"].isin(subgroup2["subID"])]
    dataset.loc[sgidx2,cThcol]=2
    sgidx3= dataset.index[dataset["subID"].isin(subgroup3["subID"])]
    dataset.loc[sgidx3,cThcol]=3
    del sgidx1, sgidx2,sgidx3
    
    return dataset



def f_test(x, y):
    x = np.array(x)
    y = np.array(y)
    f = np.var(x, ddof=1)/np.var(y, ddof=1) #calculate F test statistic 
    dfn = x.size-1 #define degrees of freedom numerator 
    dfd = y.size-1 #define degrees of freedom denominator 
    p = 1-stats.f.cdf(f, dfn, dfd) #find p-value of F test statistic 
    return f, p

def grp_data_allperms_bothsplit_plot(data,pth,stdk,figtitle,figylabel,savefilen,ymin,ymax):
    df1 = data[(data['day'] == 1)]
    df1 = df1[colcol]
    df1=pd.melt(df1) 
    df1["day"]="Day 1"
    df2 = data[(data['day'] == 2)]
    df2 = df2[colcol]
    df2=pd.melt(df2) 
    df2["day"]="Day 2"
    data=pd.concat([df1, df2], ignore_index=True)
    del df1, df2
    
    plt.figure()
    ax=sns.violinplot(data=data, x="variable",y="value", hue='day', split=True, palette=mypal3)
    sns.stripplot(data=data, x="variable",y="value", jitter=True, zorder=1, dodge=True, alpha=.1, s=2)
    plt.title(figtitle, fontsize=20)
    plt.ylabel(figylabel)
    plt.xlabel('CAP')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.ylim(ymin,ymax)
    plt.show()
    
    plt.savefig(savefilen, bbox_inches='tight')
    plt.close()
       
    return



def withinSub_betweenCap_var(inputdata,param):
    msg="\n"
    logging.info(msg)
    msg="Start plotting the distribution of within-individual between-CAP variance across permutations ...\n"
    logging.info(msg)

    # ----------- Add subgroup definitions to data ------------ #
    dataset=inputdata.copy(deep=True)       
    for cTh in param.classTh:
    
        cThstring = '_' + str(cTh) + 'nf'
        cThcol = ['subgroup' + cThstring]        
        
        dataset=add_subgroup_data_nfcluster(dataset=dataset, filein=filein, cTh=cTh)
    
        # ----------- Compute within-subject between-cap variance --------- #
        msg="Compute within-subject between-cap variance for each subject in each permutations ...\n"
        logging.info(msg)
        
        wsbcv=dataset[['0','1','2','3','4']].std(axis=1) # Normalized by N-1 by default (degree of freedom = 1)
        dataset.loc[:,"wsbcv"]=np.nan
        dataset["wsbcv"]=wsbcv         
        del wsbcv
        
        msg=str(dataset)
        logging.info(dataset)      
        
        # select data for F test between subgroups
        g1 = dataset[(dataset[cThcol[0]] == 1)] 
        g2 = dataset[(dataset[cThcol[0]] == 2)] 
        g3 = dataset[(dataset[cThcol[0]] == 3)]  
        
        # print results in log file
        # perform F-test
        f12, p12 = f_test(g1["wsbcv"], g2["wsbcv"])
        f13, p13 = f_test(g1["wsbcv"], g3["wsbcv"])
        f23, p23 = f_test(g2["wsbcv"], g3["wsbcv"])
 
        msg="\n"
        logging.info(msg)
        msg = "Performs the F test of variance between subgroups...\n"
        logging.info(msg)
        msg = cThstring + "Group 1(n=" + str(len(g1)) + ") vs Group 2(n=" + str(len(g2)) + "), F statistic=" + str(f12) + ", p=" + str(p12)
        logging.info(msg)
        msg = cThstring + "Group 1(n=" + str(len(g1)) + ") vs Group 3(n=" + str(len(g3)) + "), F statistic=" + str(f13) + ", p=" + str(p13)
        logging.info(msg)
        msg = cThstring + "Group 2(n=" + str(len(g2)) + ") vs Group 3(n=" + str(len(g3)) + "), F statistic=" + str(f23) + ", p=" + str(p23)
        logging.info(msg)        
        del f12, f13, f23, p12, p13, p23        
    
        plt.figure()
        ax=sns.violinplot(data=dataset, x=cThcol[0], y="wsbcv", hue='day', split=True, palette=mypal3)
        plt.title('WS Between-CAP Var. of FO', fontsize=20)
        plt.ylabel('WS BC Var. of FO')
        plt.xlabel('subgroup')
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        plt.ylim(0,0.2)
        plt.show()
        savefilen = filein.outdir + 'P' + str(pth) + '_Occupancy_wsbcv_allperms_bothsplit_k' + str(stdk)  + cThstring + '_trt.png'
        if savefigs:
            plt.savefig(savefilen, bbox_inches='tight')
        plt.close()  
        
        del g1, g2, g3
    
    del dataset
    
    return



def withinCAP_betweenDay_var_allperm(inputdata,datatype,param,pth,sc_ymin,sc_ymax):
    
    msg="\n"
    logging.info(msg)
    msg=datatype + ": Start plotting within-CAP between-day variance across permutations ...\n"
    logging.info(msg)

    # ----------- Add subgroup definitions to data ------------ #
    dataset=inputdata.copy(deep=True)       
    for cTh in param.classTh:
    
        cThstring = '_' + str(cTh) + 'nf'
        cThcol = ['subgroup' + cThstring]        
        
        dataset=add_subgroup_data_nfcluster(dataset=dataset, filein=filein, cTh=cTh)
 
        # ----------- Plot Within-CAP between-day variance for each CAP --------- #
        for capn in colcol:
            
            capdata = dataset[["subID", capn, cThcol[0],"day"]]
            
            #- (1 x 2) subplot for days 1 and 2 for each CAP
            plt.figure(figsize=(12, 6))
            fig, axes  = plt.subplots(1, 2,sharey='row')
            fig.suptitle('Between-day var. of ' + datatype + ': cap ' + str(capn))
            
            #- subplot 1: day 1
            daydata= capdata[capdata["day"] == 1]
            sns.violinplot(ax=axes[0], data=daydata, x=cThcol[0], y=capn, palette=["#FDE725FF","#21908CFF","#440154FF"], inner=None, linewidth=0)
            plt.setp(axes[0].collections, alpha=.5)    
            sns.boxplot(ax=axes[0], data=daydata, x=cThcol[0], y=capn, palette=["#FDE725FF","#21908CFF","#440154FF"], width=0.3, boxprops={'zorder': 2})
            axes[0].set_title('Day 1 : CAP' + str(capn))
            axes[0].set_xlabel('Day 1')
            axes[0].set_ylabel(datatype)
            axes[0].set_ylim(bottom=sc_ymin, top=sc_ymax)            
            axes[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

             #- subplot 2: day 2
            daydata= capdata[capdata["day"] == 2]
            sns.violinplot(ax=axes[1], data=daydata, x=cThcol[0], y=capn, palette=["#FDE725FF","#21908CFF","#440154FF"], inner=None, linewidth=0)
            plt.setp(axes[1].collections, alpha=.5)   
            sns.boxplot(ax=axes[1], data=daydata, x=cThcol[0], y=capn, palette=["#FDE725FF","#21908CFF","#440154FF"], width=0.3, boxprops={'zorder': 2})
            axes[1].set_title('Day 2 : CAP' + str(capn))
            axes[1].set_xlabel('Day 2')
            axes[1].set_ylabel(datatype)
            axes[1].set_ylim(bottom=sc_ymin, top=sc_ymax)            
            axes[1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
            savefilen1 = filein.outdir + 'P' + str(pth) + '_' + datatype + '_individual_betweenday_CAP' + str(capn) + cThstring + '_violinbox_new.png'
            plt.savefig(savefilen1, bbox_inches='tight')
            plt.close() 
            
        del capdata, daydata    
        
        
        # ----------- Plot Within-CAP between-day variance for each CAP --------- #
       
        day1 = dataset[dataset["day"]==1]
        day2 = dataset[dataset["day"]==2]        
        for capn in colcol:
            
            d1 = day1[["subID", capn, cThcol[0]]]
            d2 = day2[["subID", capn, cThcol[0]]]        
            
            # - plot using scatterplot
            plt.figure(figsize=(7, 7))
            # color_labels = d1[cThcol[0]].unique()
            # color_map = dict(zip(color_labels,["#FDE725FF","#21908CFF", "#440154FF"]))
            # plt.scatter(d1[capn], d2[capn], c=d1[cThcol[0]].map(color_map))
            plt.scatter(d1[capn], d2[capn], c='black', edgecolors='white',linewidth=.5)
            plt.title('Between-day changes of Individual ' + datatype + ' in CAP' + str(capn), fontsize=20)
            plt.xlabel('Day 1')
            plt.ylabel('Day 2')
            plt.xlim(sc_ymin,sc_ymax)
            plt.ylim(sc_ymin,sc_ymax)
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
            
            savefilen2 = filein.outdir + 'P' + str(pth) + '_' + datatype + '_individual_betweenday_CAP' + str(capn) + cThstring + '_scatter_new.png'           
            plt.savefig(savefilen2, bbox_inches='tight')
            plt.close()    
            
    del dataset
        
    return





def withinCAP_betweenDay_var_avgperm(inputdata,datatype,param,pth,sc_ymin,sc_ymax):
    
    msg="\n"
    logging.info(msg)
    msg=datatype + ": Start plotting within-CAP between-day variance across permutations ...\n"
    logging.info(msg)

    # ----------- Add subgroup definitions to data ------------ #
    dataset=inputdata.copy(deep=True)       
    for cTh in param.classTh:
    
        cThstring = '_' + str(cTh) + 'nf'
        cThcol = ['subgroup' + cThstring]        
        
        dataset=add_subgroup_data_nfcluster(dataset=dataset, filein=filein, cTh=cTh)
 
        # ----------- Plot Within-CAP between-day variance for each CAP --------- #
        for capn in colcol:
            
            capdata = dataset[["subID", capn, cThcol[0],"day"]]
            
            #- (1 x 2) subplot for days 1 and 2 for each CAP
            plt.figure(figsize=(12, 6))
            fig, axes  = plt.subplots(1, 2,sharey='row')
            fig.suptitle('Between-day var. of ' + datatype + ': cap ' + str(capn))
            
            #- subplot 1: day 1
            daydata= capdata[capdata["day"] == 1]
            daydata_ind = daydata.groupby("subID")[capn].mean()
            daydata_ind = daydata_ind.reset_index()
            daydata_ind = add_subgroup_data_nfcluster(dataset=daydata_ind, filein=filein, cTh=cTh)
            
            msg="day1, cap" + capn + " : " + str(daydata_ind)
            logging.info(msg)
            
            sns.violinplot(ax=axes[0], data=daydata_ind, x=cThcol[0], y=capn, palette=["#FDE725FF","#21908CFF","#440154FF"], inner=None, linewidth=0)
            plt.setp(axes[0].collections, alpha=.5)    
            sns.boxplot(ax=axes[0], data=daydata_ind, x=cThcol[0], y=capn, palette=["#FDE725FF","#21908CFF","#440154FF"], width=0.3, boxprops={'zorder': 2})
            axes[0].set_title('Day 1 : CAP' + str(capn))
            axes[0].set_xlabel('Day 1')
            axes[0].set_ylabel(datatype)
            axes[0].set_ylim(bottom=sc_ymin, top=sc_ymax)            
            axes[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
            
            del daydata, daydata_ind
            
             #- subplot 2: day 2
            daydata= capdata[capdata["day"] == 2]
            daydata_ind = daydata.groupby("subID")[capn].mean()
            daydata_ind = daydata_ind.reset_index()
            daydata_ind = add_subgroup_data_nfcluster(dataset=daydata_ind, filein=filein, cTh=cTh)
            
            msg="day2, cap" + capn + " : " + str(daydata_ind)
            logging.info(msg)
        
            sns.violinplot(ax=axes[1], data=daydata_ind, x=cThcol[0], y=capn, palette=["#FDE725FF","#21908CFF","#440154FF"], inner=None, linewidth=0)
            plt.setp(axes[1].collections, alpha=.5)   
            sns.boxplot(ax=axes[1], data=daydata_ind, x=cThcol[0], y=capn, palette=["#FDE725FF","#21908CFF","#440154FF"], width=0.3, boxprops={'zorder': 2})
            axes[1].set_title('Day 2 : CAP' + str(capn))
            axes[1].set_xlabel('Day 2')
            axes[1].set_ylabel(datatype)
            axes[1].set_ylim(bottom=sc_ymin, top=sc_ymax)            
            axes[1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
            savefilen1 = filein.outdir + 'P' + str(pth) + '_' + datatype + '_individual_betweenday_CAP' + str(capn) + cThstring + '_violinbox_avgperm_new.png'
            plt.savefig(savefilen1, bbox_inches='tight')
            plt.close() 
            
            del daydata, daydata_ind
            
        del capdata    
        
        
        # ----------- Plot Within-CAP between-day variance for each CAP --------- #
       
        day1 = dataset[dataset["day"]==1]
        day2 = dataset[dataset["day"]==2]        
        for capn in colcol:
            
            d1 = day1[["subID", capn, cThcol[0]]]
            d1_ind = d1.groupby("subID")[capn].mean()
            d1_ind = d1_ind.reset_index()
            
            d2 = day2[["subID", capn, cThcol[0]]] 
            d2_ind = d2.groupby("subID")[capn].mean()
            d2_ind = d2_ind.reset_index()    
            
            # - plot using scatterplot
            plt.figure(figsize=(7, 7))

            sns.regplot(x=d1_ind[capn], y=d2_ind[capn], color="k", line_kws={"color":"r","alpha":0.7,"lw":3})
            r, p = stats.pearsonr(x=d1_ind[capn], y=d2_ind[capn])
            plt.title('Day1vs2 ' + datatype + ', CAP' + str(capn) + '(r=' + str(r) + ', p=' + str(p) + ')', fontsize=20)
            plt.xlabel('Day 1')
            plt.ylabel('Day 2')
            plt.xlim(sc_ymin,sc_ymax)
            plt.ylim(sc_ymin,sc_ymax)
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
            
            savefilen2 = filein.outdir + 'P' + str(pth) + '_' + datatype + '_individual_betweenday_CAP' + str(capn) + cThstring + '_scatter_avgperm_new.png'           
            plt.savefig(savefilen2, bbox_inches='tight')
            plt.close()    
            
    del dataset
        
    return




def ind_dist_timemetric_eachDay_eachCap_allperm(inputdata,datatype,param,pth,xmin,xmax):
    msg="\n"
    logging.info(msg)
    msg=datatype + ": Start plotting individual " + datatype + " distribution for each CAP ...\n"
    logging.info(msg)

    for day_n in [1, 2]:
            
        for stdk in param.basis_k_range:
           
            
            for cTh in param.classTh:
            
                cThstring = '_' + str(cTh) + 'nf'
                cThcol = ['subgroup' + cThstring]
                
                df = inputdata[(inputdata['day'] == day_n)]
                dataset = df.copy(deep=True)
                dataset=add_subgroup_data_nfcluster(dataset=dataset, filein=filein, cTh=cTh)
                
                # select data for T test between subgroups
                g1 = dataset[(dataset[cThcol[0]] == 1)] 
                g2 = dataset[(dataset[cThcol[0]] == 2)] 
                g3 = dataset[(dataset[cThcol[0]] == 3)] 
                
                msg=str(g1)
                logging.info(msg)
                msg=str(g2)
                logging.info(msg)
                msg=str(g3)
                logging.info(msg)                
                
                for capn in colcol:
                
                    # print results in log file
                    t12, p12 = stats.ttest_ind(a=g1[capn], b=g2[capn])
                    t13, p13 = stats.ttest_ind(a=g1[capn], b=g3[capn])
                    t23, p23 = stats.ttest_ind(a=g2[capn], b=g3[capn])
                    # t12, p12 = stats.ks_2samp(g1[capn], g2[capn])
                    # t13, p13 = stats.ks_2samp(g1[capn], g3[capn])
                    # t23, p23 = stats.ks_2samp(g2[capn], g3[capn])    
                    msg="\n"
                    logging.info(msg)
                    # msg = "Performs the two-sided two-sample Kolmogorov-Smirnov test for goodness of fit between subgroups...\n"
                    msg = datatype+ ": Performs the two-sided two-sample T test between subgroups...\n"
                    logging.info(msg)
                    msg = cThstring + " -- CAP " +  str(capn) + ": group 1(n=" + str(len(g1)) + ") vs group 2(n=" + str(len(g2)) + "), T statistic=" + str(t12) + ", p=" + str(p12)
                    logging.info(msg)
                    msg = cThstring + " -- CAP " + str(capn) + ": group 1(n=" + str(len(g1)) + ") vs group 3(n=" + str(len(g3)) + "), T statistic=" + str(t13) + ", p=" + str(p13)
                    logging.info(msg)
                    msg = cThstring + " -- CAP " +  str(capn) + ": group 2(n=" + str(len(g2)) + ") vs group 3(n=" + str(len(g3)) + "), T statistic=" + str(t23) + ", p=" + str(p23)
                    logging.info(msg)  
                    del t12, p12, t13, p13, t23, p23
                    
                    fig = plt.figure()
                    ax = fig.add_subplot(111)   
                    sns.kdeplot(data=dataset, x=capn, hue=cThcol[0], fill=False, common_norm=False, palette=mypal2, linewidth=1)
                    sns.kdeplot(data=dataset, x=capn, hue=cThcol[0], fill=True, common_norm=False, palette=mypal,alpha=.7)
                    plt.title('Distribution of Individual ' + datatype + ' in CAP' + str(capn), fontsize=20)
                    plt.xlabel(datatype + '(CAP ' + str(capn) + ', subject j)')
                    plt.ylabel('Density')
                    plt.xlim(xmin,xmax)
                    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
                    plt.show()
                    # plt.show()
                    savefilen = filein.outdir + 'P' + str(pth) + '_' + datatype + '_individual_day' + str(day_n) + '_CAP' + str(capn) + cThstring + '_new.png'
                    if savefigs:
                        plt.savefig(savefilen, bbox_inches='tight')
                    plt.close()
                
                del dataset, g1, g2, g3, df    
    return
    


def ind_dist_timemetric_eachDay_eachCap_avgperm(inputdata,datatype,param,pth,xmin,xmax):
    msg="\n"
    logging.info(msg)
    msg=datatype + ": Start plotting individual " + datatype + " distribution for each CAP averaged over permutations ...\n"
    logging.info(msg)

    for day_n in [1, 2]:
            
        for stdk in param.basis_k_range:
           
            
            for cTh in param.classTh:
            
                cThstring = '_' + str(cTh) + 'nf'
                cThcol = ['subgroup' + cThstring]
                
                df = inputdata[(inputdata['day'] == day_n)]
                dataset = df.copy(deep=True)
                
                pval_btwgroup=np.zeros((5,3))
                tval_btwgroup=np.zeros((5,3))                
                for capn in colcol:
                    
                    capdata_ind = dataset.groupby("subID")[capn].mean()
                    capdata_ind = capdata_ind.reset_index()
                    capdata_ind=add_subgroup_data_nfcluster(dataset=capdata_ind, filein=filein, cTh=cTh)
   
                    # select data for T test between subgroups
                    g1 = capdata_ind[(capdata_ind[cThcol[0]] == 1)] 
                    g2 = capdata_ind[(capdata_ind[cThcol[0]] == 2)] 
                    g3 = capdata_ind[(capdata_ind[cThcol[0]] == 3)] 
                    
                    msg=str(g1)
                    logging.info(msg)
                    msg=str(g2)
                    logging.info(msg)
                    msg=str(g3)
                    logging.info(msg)  
                    
                    # print results in log file
                    t12, p12 = stats.ttest_ind(a=g1[capn], b=g2[capn])
                    t13, p13 = stats.ttest_ind(a=g1[capn], b=g3[capn])
                    t23, p23 = stats.ttest_ind(a=g2[capn], b=g3[capn])

                    msg="\n"
                    logging.info(msg)
                    # msg = "Performs the two-sided two-sample Kolmogorov-Smirnov test for goodness of fit between subgroups...\n"
                    msg = datatype+ ": Performs the two-sided two-sample T test between subgroups...\n"
                    logging.info(msg)
                    msg = cThstring + " -- CAP " +  str(capn) + ": group 1(n=" + str(len(g1)) + ") vs group 2(n=" + str(len(g2)) + "), T statistic=" + str(t12) + ", p=" + str(p12)
                    logging.info(msg)
                    msg = cThstring + " -- CAP " + str(capn) + ": group 1(n=" + str(len(g1)) + ") vs group 3(n=" + str(len(g3)) + "), T statistic=" + str(t13) + ", p=" + str(p13)
                    logging.info(msg)
                    msg = cThstring + " -- CAP " +  str(capn) + ": group 2(n=" + str(len(g2)) + ") vs group 3(n=" + str(len(g3)) + "), T statistic=" + str(t23) + ", p=" + str(p23)
                    logging.info(msg)  
                    
                    pval_btwgroup[int(capn),0]=p12
                    pval_btwgroup[int(capn),1]=p13
                    pval_btwgroup[int(capn),2]=p23
                    
                    tval_btwgroup[int(capn),0]=t12
                    tval_btwgroup[int(capn),1]=t13
                    tval_btwgroup[int(capn),2]=t23 
                    

                    del t12, p12, t13, p13, t23, p23
                    
                    fig = plt.figure()
                    ax = fig.add_subplot(111)   
                    sns.kdeplot(data=capdata_ind, x=capn, hue=cThcol[0], fill=False, common_norm=False, palette=mypal2, linewidth=1)
                    sns.kdeplot(data=capdata_ind, x=capn, hue=cThcol[0], fill=True, common_norm=False, palette=mypal,alpha=.7)
                    plt.title('Distribution of Individual ' + datatype + ' in CAP' + str(capn), fontsize=20)
                    plt.xlabel(datatype + '(CAP ' + str(capn) + ', subject j)')
                    plt.ylabel('Density')
                    plt.xlim(xmin,xmax)
                    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
                    plt.show()
                    # plt.show()
                    savefilen = filein.outdir + 'P' + str(pth) + '_' + datatype + '_individual_day' + str(day_n) + '_CAP' + str(capn) + cThstring + '_avgperm_new.png'
                    if savefigs:
                        plt.savefig(savefilen, bbox_inches='tight')
                    plt.close()
                    
                
                pval_btwgroup = pd.DataFrame(pval_btwgroup)
                savefilen1 = filein.outdir + 'P' + str(pth) + '_' + datatype + '_individual_day' + str(day_n) + '_allCAP' + cThstring + '_pval_btwgroup_new.csv'
                pval_btwgroup.to_csv(savefilen1)
                
                tval_btwgroup = pd.DataFrame(tval_btwgroup)
                savefilen2 = filein.outdir + 'P' + str(pth) + '_' + datatype + '_individual_day' + str(day_n) + '_allCAP' + cThstring + '_tval_btwgroup_new.csv'
                tval_btwgroup.to_csv(savefilen2)
                    
                del capdata_ind, g1, g2, g3, df    
    return
    


def inddist_bothsplit_allCaps(inputdata,datatype,pth,stdk,ymin,ymax):

    df1 = inputdata[(inputdata['day'] == 1)]
    df1 = df1[colcol]
    df1=pd.melt(df1) 
    df1["day"]="Day 1"
    df2 = inputdata[(inputdata['day'] == 2)]
    df2 = df2[colcol]
    df2=pd.melt(df2) 
    df2["day"]="Day 2"
    data=pd.concat([df1, df2], ignore_index=True)
    del df1, df2
    
    msg = datatype + " = " + str(data.shape)
    logging.info(msg)

    plt.figure()
    ax=sns.violinplot(data=data, x="variable",y="value", hue='day', split=True, palette=mypal3)
    plt.title('Individual Dwell Time Variance in each CAP', fontsize=20)
    plt.ylabel('Individual Dwell Time Variance')
    plt.xlabel('CAP')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.ylim(ymin,ymax)
    plt.show()
    savefilen = filein.outdir + 'P' + str(pth) + '_' + datatype + '_bothsplit_k' + str(stdk) + 'allCAPs_trt_new.png'
    plt.savefig(savefilen, bbox_inches='tight')

    return





def compare_to_ksolution_counts(inputdata,datatype,filein,param,pth,xmin,xmax):
    
    msg="\n"
    logging.info(msg)
    msg=datatype + ": Start plotting individual " + datatype + " distribution for each CAP vs k_solution preference ...\n"
    logging.info(msg)
    ksolution_df=pd.read_csv(filein.outdir + 'P' + str(pth) + '.0_k_solution_subcounts_allcTh.csv')

    dataset = inputdata.copy(deep=True)
    for cTh in param.classTh:

        cThstring = '_' + str(cTh) + 'nf'
        cThcol = ['subgroup' + cThstring]       
        for capn in colcol:
            
            capdata_ind = dataset.groupby("subID")[capn].mean()
            capdata_ind = capdata_ind.reset_index()
            capdata_ind=add_subgroup_data_nfcluster(dataset=capdata_ind, filein=filein, cTh=cTh)
            capdata_ind["k4-k5_counts_mean12"]=ksolution_df["k4-k5_counts_mean12"]
            
            msg=str(capdata_ind)
            logging.info(msg)
    
            plt.figure(figsize=(7, 7))
            sns.regplot(x=capdata_ind[capn], y=capdata_ind["k4-k5_counts_mean12"], color="k", line_kws={"color":"r","alpha":0.7,"lw":3})
            r, p = stats.pearsonr(x=capdata_ind[capn], y=capdata_ind["k4-k5_counts_mean12"])
            plt.title('cap' + capn + 'vs k preference (r=' + str(r) + ', p=' + str(p) + ')')
            plt.xlim(xmin, xmax)
            plt.ylim(-100, 150)
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
            savefilen = filein.outdir + 'P' + str(pth) + ".0_k4-5_subcount_vs_cap" + capn + '_' + datatype + "_scatter.png"
            plt.savefig(savefilen, bbox_inches='tight')
            plt.close() 
            
            plt.figure(figsize=(7, 7))  
            mypal=sns.color_palette("viridis_r", as_cmap=True)
            sns.scatterplot(data=capdata_ind, x=capn, y='k4-k5_counts_mean12',hue=cThcol[0], palette=mypal,  s=50, edgecolor="black")
            r, p = stats.pearsonr(x=capdata_ind[capn], y=capdata_ind["k4-k5_counts_mean12"])
            plt.title('cap' + capn + 'vs k preference (r=' + str(r) + ', p=' + str(p) + ')')
            plt.xlim(xmin, xmax)
            plt.ylim(-100, 150)
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
            savefilen2 = filein.outdir + 'P' + str(pth) + ".0_k4-5_subcount_vs_cap" + capn + '_' + datatype + "_scatter_subgroup.png"
            plt.savefig(savefilen2, bbox_inches='tight')
            plt.close()   
            
            del capdata_ind
        
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




msg="\n\n"
logging.info(msg)
msg="Visualization of time analytics starts."
logging.info(msg)
msg="\n\n"
logging.info(msg)




for pth in pthrange:

    if param.seedtype == "seedbased":
        param.sig_threshold = pth
    elif param.seedtype == "seedfree":
        param.randTthreshold = pth
        
        
    # Define variables
    colcol = ['0','1','2','3','4'] 
    mypal=sns.color_palette("viridis_r", as_cmap=True)
    mypal2=[[0,0,0],[0,0,0],[0,0,0]]    
    mypal3=[[0.5,0.5,0.5],[1,1,1]]
    colors = sns.color_palette()
    

    neworder_allperm_filen1 = filein.outdir + 'P' + str(pth) + '_split_1_SortedCAP_neworder_allperm.csv'
    neworder_allperm1 = pd.read_csv(neworder_allperm_filen1, index_col=0)  
    neworder_allperm_filen2 = filein.outdir + 'P' + str(pth) + '_split_2_SortedCAP_neworder_allperm.csv'
    neworder_allperm2 = pd.read_csv(neworder_allperm_filen2, index_col=0)      
    uniqlist1=neworder_allperm1.perm.unique()
    uniqlist2=neworder_allperm2.perm.unique()
    unique_permlist = np.intersect1d(uniqlist1,uniqlist2)
    

    # Load data

    grp_capdur_filen = filein.outdir + 'P' + str(pth) + '_Occupancy_group_trt.csv'
    grp_capdur_r_allday_allperm = pd.read_csv(grp_capdur_filen, index_col=0)
    wcbsv_dur_filen = filein.outdir + 'P' + str(pth) + '_Occupancy_wcbsv_trt.csv'
    wcbsv_dur_r_allday_allperm = pd.read_csv(wcbsv_dur_filen, index_col=0)
    wcbsm_dur_filen = filein.outdir + 'P' + str(pth) + '_Occupancy_wcbsm_trt.csv'
    wcbsm_dur_r_allday_allperm = pd.read_csv(wcbsm_dur_filen, index_col=0)
    capdur_ind_filen = filein.outdir + 'P' + str(pth) + '_Occupancy_individual_trt.csv'
    capdur_ind_allcap_r_allday_allperm = pd.read_csv(capdur_ind_filen, index_col=0)
    
    capdwt_wsm_filen = filein.outdir + 'P' + str(pth) + '_DwellTime_individual_WSmean_trt.csv'
    capdwt_wsm_allcap_r_allday_allperm = pd.read_csv(capdwt_wsm_filen, index_col=0)
    capdwt_wsv_filen = filein.outdir + 'P' + str(pth) + '_DwellTime_individual_WSstd_trt.csv'
    capdwt_wsv_allcap_r_allday_allperm = pd.read_csv(capdwt_wsv_filen, index_col=0)
    
    
    # Group level QC
    msg = "grp_capdur_r_allday_allperm " + str(grp_capdur_r_allday_allperm.shape)
    logging.info(msg)
    for day_n in [1, 2]:
            
        for stdk in param.basis_k_range:
            dataset = grp_capdur_r_allday_allperm[(grp_capdur_r_allday_allperm['day'] == day_n) & (grp_capdur_r_allday_allperm['stdk'] == stdk) ]
            msg = "[Day " + str(day_n) + "] grp_capdur_r_allday_allperm (k=" + str(stdk) +") =" + str(dataset.shape)
            logging.info(msg)
    
    # wcbsv_dur QC
    msg = "wcbsv_dur_r_allday_allperm " + str(wcbsv_dur_r_allday_allperm.shape)
    logging.info(msg)
    for day_n in [1, 2]:
            
        for stdk in param.basis_k_range:
            dataset = wcbsv_dur_r_allday_allperm[(wcbsv_dur_r_allday_allperm['day'] == day_n) & (wcbsv_dur_r_allday_allperm['stdk'] == stdk) ]
            msg = "[Day " + str(day_n) + "] wcbsv_dur_r_allday_allperm (k=" + str(stdk) +") =" + str(dataset.shape)
            logging.info(msg)
    
    # wcbsm_dur QC
    msg = "wcbsm_dur_r_allperm " + str(wcbsm_dur_r_allday_allperm.shape)
    logging.info(msg)
    for day_n in [1, 2]:
            
        for stdk in param.basis_k_range:
            dataset = wcbsm_dur_r_allday_allperm[(wcbsm_dur_r_allday_allperm['day'] == day_n) & (wcbsm_dur_r_allday_allperm['stdk'] == stdk) ]
            msg = "[Day " + str(day_n) + "] wcbsm_dur_r_allday_allperm (k=" + str(stdk) +") =" + str(dataset.shape)
            logging.info(msg) 
            
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




    # --------------------------------
    # - Add subgroup definitions to individual FO data - capdur_ind_allcap_r_allday_allperm
    # --------------------------------
    
    msg="\n"
    logging.info(msg)
    msg="Add subgroup definitions to individual FO data ...\n"
    logging.info(msg)    
    
    # ----------- Add subgroup definitions to data ------------ #
    dataset=capdur_ind_allcap_r_allday_allperm.copy(deep=True)       
    for cTh in param.classTh:
    
        cThstring = '_' + str(cTh) + 'SD'
        cThcol = ['subgroup' + cThstring]        
        
        dataset=add_subgroup_data_nfcluster(dataset=dataset, filein=filein, cTh=cTh)
    capdursg_ind_filen = filein.outdir + 'P' + str(pth) + '_Occupancy_individual_trt_subgroup.csv'    
    dataset.to_csv(capdursg_ind_filen)
 
    del dataset
 
 
    # --------------------------------
    # - Add subgroup definitions to individual mDW data - capdwt_wsm_allcap_r_allday_allperm
    # --------------------------------
    
    msg="\n"
    logging.info(msg)
    msg="Add subgroup definitions to individual mDW data ...\n"
    logging.info(msg)    
    
    # ----------- Add subgroup definitions to data ------------ #
    dataset=capdwt_wsm_allcap_r_allday_allperm.copy(deep=True)       
    for cTh in param.classTh:
    
        cThstring = '_' + str(cTh) + 'SD'
        cThcol = ['subgroup' + cThstring]        
        
        dataset=add_subgroup_data_nfcluster(dataset=dataset, filein=filein, cTh=cTh)
    capdwt_wsm_sg_ind_filen = filein.outdir + 'P' + str(pth) + '_DwellTime_individual_WSmean_trt_subgroup.csv'    
    dataset.to_csv(capdwt_wsm_sg_ind_filen)
 
    del dataset 


    # --------------------------------
    # - Add subgroup definitions to individual vDW data - capdwt_wsv_allcap_r_allday_allperm
    # --------------------------------
    
    msg="\n"
    logging.info(msg)
    msg="Add subgroup definitions to individual vDW data ...\n"
    logging.info(msg)    
    
    # ----------- Add subgroup definitions to data ------------ #
    dataset=capdwt_wsv_allcap_r_allday_allperm.copy(deep=True)       
    for cTh in param.classTh:
    
        cThstring = '_' + str(cTh) + 'SD'
        cThcol = ['subgroup' + cThstring]        
        
        dataset=add_subgroup_data_nfcluster(dataset=dataset, filein=filein, cTh=cTh)
    capdwt_wsm_sg_ind_filen = filein.outdir + 'P' + str(pth) + '_DwellTime_individual_WSstd_trt_subgroup.csv'    
    dataset.to_csv(capdwt_wsm_sg_ind_filen)
 
    del dataset 


    # -------------------------------------------------------------
    # 
    #                           PLOT DATA 
    # 
    # -------------------------------------------------------------
    
   
    # --------------------------------
    # - grp_capdur_r_allday_allperm
    # --------------------------------
    figtitle = '(Sum of subj) Total Occupancy in each CAP'
    figylabel = '(Sum of subj)  Total Occupancy'
    savefilen = filein.outdir + 'P' + str(pth) + '_Occupancy_grp_capdur_allperms_bothsplit_k' + str(stdk) + 'allCAPs_trt_new.png'
    grp_data_allperms_bothsplit_plot(data=grp_capdur_r_allday_allperm,pth=pth,stdk=stdk,figtitle=figtitle,figylabel=figylabel,savefilen=savefilen,ymin=0,ymax=40)

 
    # --------------------------------
    # - wcbsm_dur_r_allday_allperm
    # --------------------------------
    figtitle = 'Mean of Individual Occupancy in each CAP'
    figylabel = 'Mean of Individual Occupancy'
    savefilen = filein.outdir + 'P' + str(pth) + '_Occupancy_wcbsm_dur_allperms_bothsplit_k' + str(stdk) + 'allCAPs_trt_new.png'
    grp_data_allperms_bothsplit_plot(data=wcbsm_dur_r_allday_allperm,pth=pth,stdk=stdk,figtitle=figtitle,figylabel=figylabel,savefilen=savefilen,ymin=0,ymax=0.3)


    # --------------------------------
    # - wcbsv_dur_r_allday_allperm
    # --------------------------------
    figtitle = 'Variance of Individual Occupancy in each CAP'
    figylabel = 'Variance of Individual Occupancy'
    savefilen = filein.outdir + 'P' + str(pth) + '_Occupancy_wcbsv_dur_allperms_bothsplit_k' + str(stdk) + 'allCAPs_trt_new.png'
    grp_data_allperms_bothsplit_plot(data=wcbsv_dur_r_allday_allperm,pth=pth,stdk=stdk,figtitle=figtitle,figylabel=figylabel,savefilen=savefilen,ymin=0,ymax=0.5)



    # --------------------------------
    # - Within-individual between-CAP variance - capdur_ind_allcap_r_allday_allperm
    # --------------------------------
    withinSub_betweenCap_var(inputdata=capdur_ind_allcap_r_allday_allperm,param=param)
    

    # --------------------------------
    # - Within-CAP between-day variance - all perm
    # --------------------------------
    
    # - capdur_ind_allcap_r_allday_allperm
    withinCAP_betweenDay_var_allperm(inputdata=capdur_ind_allcap_r_allday_allperm,datatype='Occupancy',param=param,pth=pth,sc_ymin=0,sc_ymax=0.4)
 
    # - capdwt_wsv_allcap_r_allday_allperm
    withinCAP_betweenDay_var_allperm(inputdata=capdwt_wsv_allcap_r_allday_allperm,datatype='DwellTime_WSstd',param=param,pth=pth,sc_ymin=0,sc_ymax=12)
 
    # - capdwt_wsm_allcap_r_allday_allperm
    withinCAP_betweenDay_var_allperm(inputdata=capdwt_wsm_allcap_r_allday_allperm,datatype='DwellTime_WSmean',param=param,pth=pth,sc_ymin=0,sc_ymax=8)
 

    # --------------------------------
    # - Within-CAP between-day variance - average over perm
    # --------------------------------
    
    # - capdur_ind_allcap_r_allday_allperm
    withinCAP_betweenDay_var_avgperm(inputdata=capdur_ind_allcap_r_allday_allperm,datatype='Occupancy',param=param,pth=pth,sc_ymin=0,sc_ymax=0.25)
 
    # - capdwt_wsv_allcap_r_allday_allperm
    withinCAP_betweenDay_var_avgperm(inputdata=capdwt_wsv_allcap_r_allday_allperm,datatype='DwellTime_WSstd',param=param,pth=pth,sc_ymin=0,sc_ymax=10)
 
    # - capdwt_wsm_allcap_r_allday_allperm
    withinCAP_betweenDay_var_avgperm(inputdata=capdwt_wsm_allcap_r_allday_allperm,datatype='DwellTime_WSmean',param=param,pth=pth,sc_ymin=0,sc_ymax=7)
 
    
    
    # --------------------------------
    # - Individual distribution for each CAP - all perm
    # --------------------------------
    
    # - capdur_ind_allcap_r_allday_allperm
    ind_dist_timemetric_eachDay_eachCap_allperm(inputdata=capdur_ind_allcap_r_allday_allperm,datatype='Occupancy',param=param,pth=pth,xmin=0,xmax=0.4)

    # - capdwt_wsv_allcap_r_allday_allperm
    ind_dist_timemetric_eachDay_eachCap_allperm(inputdata=capdwt_wsv_allcap_r_allday_allperm,datatype='DwellTime_WSstd',param=param,pth=pth,xmin=0,xmax=10)
    
    # - capdwt_wsm_allcap_r_allday_allperm
    ind_dist_timemetric_eachDay_eachCap_allperm(inputdata=capdwt_wsm_allcap_r_allday_allperm,datatype='DwellTime_WSmean',param=param,pth=pth,xmin=0,xmax=7)



    # --------------------------------
    # - Individual distribution for each CAP - average over perm
    # --------------------------------
    
    # - capdur_ind_allcap_r_allday_allperm
    ind_dist_timemetric_eachDay_eachCap_avgperm(inputdata=capdur_ind_allcap_r_allday_allperm,datatype='Occupancy',param=param,pth=pth,xmin=0,xmax=0.25)

    # - capdwt_wsv_allcap_r_allday_allperm
    ind_dist_timemetric_eachDay_eachCap_avgperm(inputdata=capdwt_wsv_allcap_r_allday_allperm,datatype='DwellTime_WSstd',param=param,pth=pth,xmin=0,xmax=10)
    
    # - capdwt_wsm_allcap_r_allday_allperm
    ind_dist_timemetric_eachDay_eachCap_avgperm(inputdata=capdwt_wsm_allcap_r_allday_allperm,datatype='DwellTime_WSmean',param=param,pth=pth,xmin=0,xmax=7)



    # --------------------------------
    # Individual distribution bothsplit all CAPs - capdwt_wsv_allcap_r_allday_allperm
    # --------------------------------
    inddist_bothsplit_allCaps(inputdata=capdwt_wsv_allcap_r_allday_allperm,datatype='DwellTime_WithinSubSTD',pth=pth,stdk=stdk,ymin=0,ymax=12)



    # --------------------------------
    # - Compare to k_solution counts
    # --------------------------------
    
    # - capdur_ind_allcap_r_allday_allperm
    compare_to_ksolution_counts(inputdata=capdur_ind_allcap_r_allday_allperm,datatype='Occupancy',filein=filein,param=param,pth=pth,xmin=0,xmax=0.25)
    
    # - capdwt_wsv_allcap_r_allday_allperm
    compare_to_ksolution_counts(inputdata=capdwt_wsv_allcap_r_allday_allperm,datatype='DwellTime_WSstd',filein=filein,param=param,pth=pth,xmin=0,xmax=10)

    # - capdwt_wsm_allcap_r_allday_allperm
    compare_to_ksolution_counts(inputdata=capdwt_wsm_allcap_r_allday_allperm,datatype='DwellTime_WSmean',filein=filein,param=param,pth=pth,xmin=0,xmax=7)



    # # --------------------------------
    # # - Compute ICC (Fractional occupancy)for each cap and avg over perm - capdur_ind_allcap_r_allday_allperm
    # # - input data = 168 subjects x 2 days
    # # --------------------------------

    
    # msg="\n"
    # logging.info(msg)
    # msg="Compute ICC of individual FO values for each CAP over 2 days ...\n"
    # logging.info(msg)

    
    # for stdk in param.basis_k_range:
        
    #     for capn in colcol:
            
    #         for sp in [1, 2]:
                
    #             if sp == 1:
    #                 sptitletag = "split_1"
    #             elif sp == 2:
    #                 sptitletag = "split_2"    
                
    #             neworder_allperm_filen = filein.outdir + 'P' + str(pth) + "_" + sptitletag + '_SortedCAP_neworder_allperm.csv'
    #             neworder_allperm = pd.read_csv(neworder_allperm_filen, index_col=0)  

    #             for index in range(len(neworder_allperm)):
    
    #                 perm = neworder_allperm["perm"][index]
    #                 data = capdur_ind_allcap_r_allday_allperm[(capdur_ind_allcap_r_allday_allperm['permidx'] == perm) & (capdur_ind_allcap_r_allday_allperm['half'] == sptitletag)]

    #                 msg="CAP " + str(capn) + " --- " + sptitletag + ": perm " + str(perm)
    #                 logging.info(msg)
                    
    #                 # Compute ICC for test-retest reliability. Take ICC2: Two-way random effects model
    #                 import pingouin as pg
    #                 stats = pg.intraclass_corr(data=data, targets='subID', raters='day', ratings=capn).round(3)
    #                 icc = stats[stats['Type'] == 'ICC2']
    #                 icc = icc["ICC"]

    #                 df_warg = {'n_cap': capn, 'half': sptitletag, 'perm': perm, 'ICC': icc}
    #                 col_warg = ['n_cap','half','perm','ICC']
    #                 icc_df = pd.DataFrame(df_warg, columns=col_warg)
                    
    #                 if 'icc_df_allperm' in locals():
    #                     icc_df_allperm = pd.concat([icc_df_allperm, icc_df], axis=0, ignore_index=True)
    #                 else:
    #                     icc_df_allperm = icc_df                    
                
    # # Save results
    # icc_capdur_filen = filein.outdir + 'P' + str(pth) + '_Occupancy_individual_ICC.csv'
    # icc_df_allperm.to_csv(icc_capdur_filen)      


    # --------------------------------
    # - Avg ICC (fractional occupancy) for each cap over perm - capdur_ind_allcap_r_allday_allperm
    # - plot
    # --------------------------------

    icc_capdur_filen = filein.outdir + 'P' + str(pth) + '_Occupancy_individual_ICC.csv'
    icc_df_allperm = pd.read_csv(icc_capdur_filen, index_col=0)
    # Fill zero occupancy for 5th CAP 
    # icc_df_allperm['ICC'] = icc_df_allperm['ICC'].fillna(0)

    msg=str(icc_df_allperm)
    logging.info(msg)
    
    icc_mean=icc_df_allperm.groupby(['n_cap'])['ICC'].mean()
    icc_std=icc_df_allperm.groupby(['n_cap'])['ICC'].std()
    
    msg = "Occupancy: mean - " + str(icc_mean) + " std " + str(icc_std)
    logging.info(msg)
    
    icc_all_mean=icc_df_allperm['ICC'].mean()
    icc_all_std=icc_df_allperm['ICC'].std()
    
    msg = "Occupancy (all): mean - " + str(icc_all_mean) + " std " + str(icc_all_std)
    logging.info(msg)
    
    plt.figure()
    sns.violinplot(data=icc_df_allperm, x="n_cap",y="ICC", color=[0.5,0.5,0.5])
    plt.title('ICC of individual FO in each CAP', fontsize=20)
    plt.ylabel('ICC2(FO)')
    plt.xlabel('CAP')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.ylim(0,1)
    plt.show()
    savefilen = filein.outdir + 'P' + str(pth) + '_Occupancy_individual_k' + str(stdk) + 'allCAPs_ICC.png'
    if savefigs:
        plt.savefig(savefilen, bbox_inches='tight')
    plt.close()  

    del icc_df_allperm, icc_mean, icc_std, icc_all_mean, icc_all_std




    # # --------------------------------
    # # - Compute ICC (mean dwell time)for each cap and avg over perm - capdwt_wsm_allcap_r_allday_allperm
    # # - input data = 168 subjects x 2 days
    # # --------------------------------

    
    # msg="\n"
    # logging.info(msg)
    # msg="Compute ICC of individual mean DT values for each CAP over 2 days ...\n"
    # logging.info(msg)

    
    # for stdk in param.basis_k_range:
        
    #     for capn in colcol:
            
    #         for sp in [1, 2]:
                
    #             if sp == 1:
    #                 sptitletag = "split_1"
    #             elif sp == 2:
    #                 sptitletag = "split_2"    
                
    #             neworder_allperm_filen = filein.outdir + 'P' + str(pth) + "_" + sptitletag + '_SortedCAP_neworder_allperm.csv'
    #             neworder_allperm = pd.read_csv(neworder_allperm_filen, index_col=0)  

    #             for index in range(len(neworder_allperm)):
    
    #                 perm = neworder_allperm["perm"][index]
    #                 data = capdwt_wsm_allcap_r_allday_allperm[(capdwt_wsm_allcap_r_allday_allperm['permidx'] == perm) & (capdwt_wsm_allcap_r_allday_allperm['half'] == sptitletag)]

    #                 msg="CAP " + str(capn) + " --- " + sptitletag + ": perm " + str(perm) + str(data.shape)
    #                 logging.info(msg)
                    
    #                 # Compute ICC for test-retest reliability. Take ICC2: Two-way random effects model
    #                 import pingouin as pg
    #                 stats = pg.intraclass_corr(data=data, targets='subID', raters='day', ratings=capn).round(3)
    #                 icc = stats[stats['Type'] == 'ICC2']
    #                 icc = icc["ICC"]

    #                 df_warg = {'n_cap': capn, 'half': sptitletag, 'perm': perm, 'ICC': icc}
    #                 col_warg = ['n_cap','half','perm','ICC']
    #                 icc_df = pd.DataFrame(df_warg, columns=col_warg)
                    
    #                 if 'icc_df_allperm' in locals():
    #                     icc_df_allperm = pd.concat([icc_df_allperm, icc_df], axis=0, ignore_index=True)
    #                 else:
    #                     icc_df_allperm = icc_df                    
                
    # # Save results
    # icc_capdwt_filen = filein.outdir + 'P' + str(pth) + '_DwellTime_individual_ICC.csv'
    # icc_df_allperm.to_csv(icc_capdwt_filen)      



    # --------------------------------
    # - Avg ICC (mean dwell time) for each cap over perm - capdwt_wsm_allcap_r_allday_allperm
    # - plot
    # --------------------------------

    icc_capdwt_filen = filein.outdir + 'P' + str(pth) + '_DwellTime_individual_ICC.csv'
    icc_df_allperm = pd.read_csv(icc_capdwt_filen, index_col=0)
    # Fill zero mean dwell time for 5th CAP 
    # icc_df_allperm['ICC'] = icc_df_allperm['ICC'].fillna(0)


    icc_mean=icc_df_allperm.groupby(['n_cap'])['ICC'].mean()
    icc_std=icc_df_allperm.groupby(['n_cap'])['ICC'].std()
    
    msg = "mean DT: mean - " + str(icc_mean) + " std " + str(icc_std)
    logging.info(msg)


    icc_all_mean=icc_df_allperm['ICC'].mean()
    icc_all_std=icc_df_allperm['ICC'].std()
    
    msg = "mean DT (all): mean - " + str(icc_all_mean) + " std " + str(icc_all_std)
    logging.info(msg)

    plt.figure()
    sns.violinplot(data=icc_df_allperm, x="n_cap",y="ICC", color=[0.5,0.5,0.5])
    plt.title('ICC of individual mean DT in each CAP', fontsize=20)
    plt.ylabel('ICC2(mean DT)')
    plt.xlabel('CAP')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.ylim(0,1)
    plt.show()
    savefilen = filein.outdir + 'P' + str(pth) + '_DwellTime_individual_k' + str(stdk) + 'allCAPs_ICC.png'
    if savefigs:
        plt.savefig(savefilen, bbox_inches='tight')
    plt.close()  

    del icc_df_allperm, icc_mean, icc_std, icc_all_mean, icc_all_std





    # # --------------------------------
    # # - Compute ICC (variance dwell time)for each cap and avg over perm - capdwt_wsv_allcap_r_allday_allperm
    # # - input data = 168 subjects x 2 days
    # # --------------------------------

    
    # msg="\n"
    # logging.info(msg)
    # msg="Compute ICC of individual DT variance values for each CAP over 2 days ...\n"
    # logging.info(msg)

    
    # for stdk in param.basis_k_range:
        
    #     for capn in colcol:
            
    #         for sp in [1, 2]:
                
    #             if sp == 1:
    #                 sptitletag = "split_1"
    #             elif sp == 2:
    #                 sptitletag = "split_2"    
                
    #             neworder_allperm_filen = filein.outdir + 'P' + str(pth) + "_" + sptitletag + '_SortedCAP_neworder_allperm.csv'
    #             neworder_allperm = pd.read_csv(neworder_allperm_filen, index_col=0)  

    #             for index in range(len(neworder_allperm)):
    
    #                 perm = neworder_allperm["perm"][index]
    #                 data = capdwt_wsv_allcap_r_allday_allperm[(capdwt_wsv_allcap_r_allday_allperm['permidx'] == perm) & (capdwt_wsv_allcap_r_allday_allperm['half'] == sptitletag)]

    #                 msg="CAP " + str(capn) + " --- " + sptitletag + ": perm " + str(perm) + str(data.shape)
    #                 logging.info(msg)
                    
    #                 # Compute ICC for test-retest reliability. Take ICC2: Two-way random effects model
    #                 import pingouin as pg
    #                 stats = pg.intraclass_corr(data=data, targets='subID', raters='day', ratings=capn).round(3)
    #                 icc = stats[stats['Type'] == 'ICC2']
    #                 icc = icc["ICC"]

    #                 df_warg = {'n_cap': capn, 'half': sptitletag, 'perm': perm, 'ICC': icc}
    #                 col_warg = ['n_cap','half','perm','ICC']
    #                 icc_df = pd.DataFrame(df_warg, columns=col_warg)
                    
    #                 if 'icc_df_allperm' in locals():
    #                     icc_df_allperm = pd.concat([icc_df_allperm, icc_df], axis=0, ignore_index=True)
    #                 else:
    #                     icc_df_allperm = icc_df                    
                
    # # Save results
    # icc_capdwt_filen = filein.outdir + 'P' + str(pth) + '_DwellTime_WSstd_individual_ICC.csv'
    # icc_df_allperm.to_csv(icc_capdwt_filen)      



    # --------------------------------
    # - Avg ICC (mean dwell time) for each cap over perm - capdwt_wsm_allcap_r_allday_allperm
    # - plot
    # --------------------------------

    icc_capdwt_filen = filein.outdir + 'P' + str(pth) + '_DwellTime_WSstd_individual_ICC.csv'
    icc_df_allperm = pd.read_csv(icc_capdwt_filen, index_col=0)
    # Fill zero mean dwell time for 5th CAP 
    # icc_df_allperm['ICC'] = icc_df_allperm['ICC'].fillna(0)


    icc_mean=icc_df_allperm.groupby(['n_cap'])['ICC'].mean()
    icc_std=icc_df_allperm.groupby(['n_cap'])['ICC'].std()
    
    msg = "DT var: mean - " + str(icc_mean) + " std " + str(icc_std)
    logging.info(msg)

    icc_all_mean=icc_df_allperm['ICC'].mean()
    icc_all_std=icc_df_allperm['ICC'].std()
    
    msg = "var DT (all): mean - " + str(icc_all_mean) + " std " + str(icc_all_std)
    logging.info(msg)



    plt.figure()
    sns.violinplot(data=icc_df_allperm, x="n_cap",y="ICC", color=[0.5,0.5,0.5])
    plt.title('ICC of individual DT variance in each CAP', fontsize=20)
    plt.ylabel('ICC2(DT var)')
    plt.xlabel('CAP')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.ylim(0,1)
    plt.show()
    savefilen = filein.outdir + 'P' + str(pth) + '_DwellTime_WSstd_individual_k' + str(stdk) + 'allCAPs_ICC.png'
    if savefigs:
        plt.savefig(savefilen, bbox_inches='tight')
    plt.close()  

    del icc_df_allperm, icc_mean, icc_std, icc_all_mean, icc_all_std





    # # --------------------------------
    # # - 2-axis plot - mean individuals (both days projected together)
    # # --------------------------------
    # #- axis 1: between-subject variance of Occupancy (estimated from the whole 4 runs) ~2000 permutations from split 1 and 2 
    # wcbsv_dur_filen = filein.outdir + 'P' + str(pth) + '_Occupancy_wcbsv_trt.csv'
    # wcbsv_dur = pd.read_csv(wcbsv_dur_filen, index_col=0)  
    # #- axis 2: within-subject variance of Dwell-time (estimated from the whole 4 runs) ~2000 permutations from split 1 and 2 x 168 subjects 
    # wcwsv_dwt_filen = filein.outdir + 'P' + str(pth) + '_DwellTime_individual_WSstd_trt.csv'
    # wcwsv_dwt = pd.read_csv(wcwsv_dwt_filen, index_col=0)    
    
    # msg = "axis-1 = " + str(wcbsv_dur.shape) + ", axis-2 = " + str(wcwsv_dwt.shape)
    # logging.info(msg)
    
    
    # for testcol in colcol:
        
    #     fig = plt.figure(figsize = (32, 32))

    #     for capn in testcol:
                            
    #         for perm in unique_permlist:
                
                
    #             ##----------------------------------------------------------------
    #             # load and reshape data
    #             ##----------------------------------------------------------------
                
    #             sp1_ax1 = wcbsv_dur[(wcbsv_dur['permidx'] == perm) & (wcbsv_dur['half'] == "split_1")]
    #             sp1_ax1 = sp1_ax1[capn]
                
    #             sp1_day1_ax2 = wcwsv_dwt[(wcwsv_dwt['permidx'] == perm) & (wcwsv_dwt['half'] == "split_1") & (wcwsv_dwt['day'] == 1)]
    #             sp1_day2_ax2 = wcwsv_dwt[(wcwsv_dwt['permidx'] == perm) & (wcwsv_dwt['half'] == "split_1") & (wcwsv_dwt['day'] == 2)]
    #             a1 = sp1_day1_ax2[capn].mean()
    #             a2 = sp1_day2_ax2[capn].mean()
    #             df=np.vstack((a1, a2)).squeeze(axis=1)
    #             sp1_ax2 = pd.DataFrame(df, columns = [capn])
    #             del a1, a2, df
                

    #             sp2_ax1 = wcbsv_dur[(wcbsv_dur['permidx'] == perm) & (wcbsv_dur['half'] == "split_2")]
    #             sp2_ax1 = sp2_ax1[capn]
                
    #             sp2_day1_ax2 = wcwsv_dwt[(wcwsv_dwt['permidx'] == perm) & (wcwsv_dwt['half'] == "split_2") & (wcwsv_dwt['day'] == 1)]
    #             sp2_day2_ax2 = wcwsv_dwt[(wcwsv_dwt['permidx'] == perm) & (wcwsv_dwt['half'] == "split_2") & (wcwsv_dwt['day'] == 2)]
    #             a1 = sp2_day1_ax2[capn].mean()
    #             a2 = sp2_day2_ax2[capn].mean()
    #             df=np.vstack((a1, a2)).squeeze(axis=1)
    #             sp2_ax2 = pd.DataFrame(df, columns = [capn])
    #             del a1, a2, df
                                
    #             ax1 = pd.concat([sp1_ax1,sp2_ax1], axis=0, ignore_index=True)
    #             ax2 = pd.concat([sp1_ax2,sp2_ax2], axis=0, ignore_index=True)

 
                
    #             del sp1_ax1, sp1_ax2, sp2_ax1, sp1_day1_ax2, sp1_day2_ax2, sp2_day1_ax2, sp2_day2_ax2
                
                
    #             ##----------------------------------------------------------------
    #             # Collect across permutations
    #             ##----------------------------------------------------------------
                    
    #             if 'allperm_ax1' in locals():
    #                 allperm_ax1=pd.concat([allperm_ax1, ax1], ignore_index=True)
    #             else:
    #                 allperm_ax1=ax1
                    
    #             if 'allperm_ax2' in locals():
    #                 allperm_ax2=pd.concat([allperm_ax2, ax2], ignore_index=True)
    #             else:
    #                 allperm_ax2=ax2
                    

                
    #             msg = "CAP " + str(capn) + " perm " + str(perm) + ": ax1 = " + str(allperm_ax1.shape) + ", ax2 = " + str(allperm_ax2.shape) 
    #             logging.info(msg)                
    #             del ax1, ax2
            
    
    #         ##----------------------------------------------------------------
    #         #  Display for each CAP
    #         ##----------------------------------------------------------------
    #         palette_tab10 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    #         pal=[palette_tab10[int(capn)]] * allperm_ax1.shape[0]
    #         plt.scatter(allperm_ax1, allperm_ax2, marker='o', s=1000,c=pal, alpha=.1)                   
    #         plt.title("3D scatter plot - Time Metrics")
    #         plt.xlabel('BSV(FO)', fontweight ='bold')
    #         plt.ylabel('WSV(DT)', fontweight ='bold')
    #         plt.xlim(0,0.05)
    #         plt.ylim(2,4.5)
    #         plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)


            
    #     # plt.show()
    #     savefilen = filein.outdir + 'P' + str(pth) + '_2dScatter_BSV_FO_WSV_DT_cap' + str(capn) + '.png'
    #     plt.savefig(savefilen, bbox_inches='tight')
    #     plt.close()
        
    #     del allperm_ax1, allperm_ax2, pal





    # # --------------------------------
    # # - 2-axis plot - mean individuals - all caps overlap (caps 1/2 together, 3/4 together, and 5) (both days projected together)
    # # --------------------------------
    # #- axis 1: between-subject variance of Occupancy (estimated from the whole 4 runs) ~2000 permutations from split 1 and 2 
    # wcbsv_dur_filen = filein.outdir + 'P' + str(pth) + '_Occupancy_wcbsv_trt.csv'
    # wcbsv_dur = pd.read_csv(wcbsv_dur_filen, index_col=0)  
    # #- axis 2: within-subject variance of Dwell-time (estimated from the whole 4 runs) ~2000 permutations from split 1 and 2 x 168 subjects 
    # wcwsv_dwt_filen = filein.outdir + 'P' + str(pth) + '_DwellTime_individual_WSstd_trt.csv'
    # wcwsv_dwt = pd.read_csv(wcwsv_dwt_filen, index_col=0)    
    
    # msg = "axis-1 = " + str(wcbsv_dur.shape) + ", axis-2 = " + str(wcwsv_dwt.shape)
    # logging.info(msg)
    
    # fig = plt.figure(figsize = (32, 32))
    
    # for testcol in colcol:
        
    #     for capn in testcol:
                            
    #         for perm in unique_permlist:
                
                
    #             ##----------------------------------------------------------------
    #             # load and reshape data
    #             ##----------------------------------------------------------------
                
    #             sp1_ax1 = wcbsv_dur[(wcbsv_dur['permidx'] == perm) & (wcbsv_dur['half'] == "split_1")]
    #             sp1_ax1 = sp1_ax1[capn]
                
    #             sp1_day1_ax2 = wcwsv_dwt[(wcwsv_dwt['permidx'] == perm) & (wcwsv_dwt['half'] == "split_1") & (wcwsv_dwt['day'] == 1)]
    #             sp1_day2_ax2 = wcwsv_dwt[(wcwsv_dwt['permidx'] == perm) & (wcwsv_dwt['half'] == "split_1") & (wcwsv_dwt['day'] == 2)]
    #             a1 = sp1_day1_ax2[capn].mean()
    #             a2 = sp1_day2_ax2[capn].mean()
    #             df=np.vstack((a1, a2)).squeeze(axis=1)
    #             sp1_ax2 = pd.DataFrame(df, columns = [capn])
    #             del a1, a2, df
                

    #             sp2_ax1 = wcbsv_dur[(wcbsv_dur['permidx'] == perm) & (wcbsv_dur['half'] == "split_2")]
    #             sp2_ax1 = sp2_ax1[capn]
                
    #             sp2_day1_ax2 = wcwsv_dwt[(wcwsv_dwt['permidx'] == perm) & (wcwsv_dwt['half'] == "split_2") & (wcwsv_dwt['day'] == 1)]
    #             sp2_day2_ax2 = wcwsv_dwt[(wcwsv_dwt['permidx'] == perm) & (wcwsv_dwt['half'] == "split_2") & (wcwsv_dwt['day'] == 2)]
    #             a1 = sp2_day1_ax2[capn].mean()
    #             a2 = sp2_day2_ax2[capn].mean()
    #             df=np.vstack((a1, a2)).squeeze(axis=1)
    #             sp2_ax2 = pd.DataFrame(df, columns = [capn])
    #             del a1, a2, df
                                
    #             ax1 = pd.concat([sp1_ax1,sp2_ax1], axis=0, ignore_index=True)
    #             ax2 = pd.concat([sp1_ax2,sp2_ax2], axis=0, ignore_index=True)

 
                
    #             del sp1_ax1, sp1_ax2, sp2_ax1, sp1_day1_ax2, sp1_day2_ax2, sp2_day1_ax2, sp2_day2_ax2
                
                
    #             ##----------------------------------------------------------------
    #             # Collect across permutations
    #             ##----------------------------------------------------------------
                    
    #             if 'allperm_ax1' in locals():
    #                 allperm_ax1=pd.concat([allperm_ax1, ax1], ignore_index=True)
    #             else:
    #                 allperm_ax1=ax1
                    
    #             if 'allperm_ax2' in locals():
    #                 allperm_ax2=pd.concat([allperm_ax2, ax2], ignore_index=True)
    #             else:
    #                 allperm_ax2=ax2
                    

                
    #             msg = "CAP " + str(capn) + " perm " + str(perm) + ": ax1 = " + str(allperm_ax1.shape) + ", ax2 = " + str(allperm_ax2.shape) 
    #             logging.info(msg)                
    #             del ax1, ax2
            
    
    #         ##----------------------------------------------------------------
    #         #  Display for each CAP
    #         ##----------------------------------------------------------------
    #         palette_tab10 = ['#ff7f0e', '#ff7f0e', '#2ca02c', '#2ca02c', '#9467bd']
    #         pal=[palette_tab10[int(capn)]] * allperm_ax1.shape[0]
    #         plt.scatter(allperm_ax1, allperm_ax2, marker='o', s=1000,c=pal, alpha=.1)                   
    #         plt.title("3D scatter plot - Time Metrics")
    #         plt.xlabel('BSV(FO)', fontweight ='bold')
    #         plt.ylabel('WSV(DT)', fontweight ='bold')
    #         plt.xlim(0,0.05)
    #         plt.ylim(2,4.5)
    #         plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

        
    #     del allperm_ax1, allperm_ax2, pal
    
    # # plt.show()
    # savefilen = filein.outdir + 'P' + str(pth) + '_2dScatter_BSV_FO_WSV_DT_allcap.png'
    # plt.savefig(savefilen, bbox_inches='tight')
    # plt.close()











    # # --------------------------------
    # # - 2-axis plot - individual (both days projected together)
    # # --------------------------------
    
    
    # msg="\n"
    # logging.info(msg)
    # msg="Start plotting individual 2d projection for each CAP ...\n"
    # logging.info(msg)

        
    # for stdk in param.basis_k_range:
       
        
    #     for cTh in param.classTh:
        
    #         cThstring = '_' + str(cTh) + 'nf'
    #         cThcol = ['subgroup' + cThstring]
            
        
    #         # Load data
    #         #- axis 1: between-subject variance of Occupancy (estimated from the whole 4 runs) ~2000 permutations from split 1 and 2
    #         wcbsv_dur_filen = filein.outdir + 'P' + str(pth) + '_Occupancy_wcbsv_trt.csv'
    #         wcbsv_dur = pd.read_csv(wcbsv_dur_filen, index_col=0)  
    #         #- axis 2: within-subject variance of Dwell-time (estimated from the whole 4 runs) ~2000 permutations from split 1 and 2 x 168 subjects 
    #         wcwsv_dwt_filen = filein.outdir + 'P' + str(pth) + '_DwellTime_individual_WSstd_trt.csv'
    #         wcwsv_dwt = pd.read_csv(wcwsv_dwt_filen, index_col=0)  
    #         wcwsv_dwt=add_subgroup_data_nfcluster(dataset=wcwsv_dwt, filein=filein, cTh=cTh)
            
 
    #         # -------------------------------------------
    #         #          Select CAP to plot
    #         # -------------------------------------------
 
    #         for testcol in colcol:
                
    #             fig = plt.figure(figsize = (32, 32))
        
    #             for capn in testcol:
                                    
    #                 for perm in unique_permlist:
                        
                        
    #                     ##----------------------------------------------------------------
    #                     # load and reshape data
    #                     ##----------------------------------------------------------------
                        
    #                     sp1_ax1 = wcbsv_dur[(wcbsv_dur['permidx'] == perm) & (wcbsv_dur['half'] == "split_1")]
    #                     sp1_ax2 = wcwsv_dwt[(wcwsv_dwt['permidx'] == perm) & (wcwsv_dwt['half'] == "split_1")]

    #                     # msg = "split1_ax1 : " + str(sp1_ax1) + "\n ax2 : " + str(sp1_ax2)
    #                     # logging.info(msg)
        
    #                     sp1_ax1 = sp1_ax1[capn] 
    #                     sp1_ax1 = sp1_ax1.iloc[np.arange(len(sp1_ax1)).repeat(sp1_ax2.shape[0])]
    #                     sp1_ax2 = sp1_ax2[[capn,cThcol[0]]]
                        

                        
    #                     sp2_ax1 = wcbsv_dur[(wcbsv_dur['permidx'] == perm) & (wcbsv_dur['half'] == "split_2")]
    #                     sp2_ax2 = wcwsv_dwt[(wcwsv_dwt['permidx'] == perm) & (wcwsv_dwt['half'] == "split_2")]
        
    #                     sp2_ax1 = sp2_ax1[capn]
    #                     sp2_ax1 = sp2_ax1.iloc[np.arange(len(sp2_ax1)).repeat(sp2_ax2.shape[0])]
    #                     sp2_ax2 = sp2_ax2[[capn,cThcol[0]]]

    #                     # msg = "split2_ax1 : " + str(sp2_ax1) + "\n ax2 : " + str(sp2_ax2)
    #                     # logging.info(msg)                        

    #                     ax1 = pd.concat([sp1_ax1,sp2_ax1], axis=0, ignore_index=True)
    #                     ax2 = pd.concat([sp1_ax2,sp2_ax2], axis=0, ignore_index=True)
         
                        
    #                     del sp1_ax1, sp1_ax2, sp2_ax1, sp2_ax2
                        
                        
    #                     ##----------------------------------------------------------------
    #                     # Collect across permutations
    #                     ##----------------------------------------------------------------
                            
    #                     if 'allperm_ax1' in locals():
    #                         allperm_ax1=pd.concat([allperm_ax1, ax1], ignore_index=True)
    #                     else:
    #                         allperm_ax1=ax1
                            
    #                     if 'allperm_ax2' in locals():
    #                         allperm_ax2=pd.concat([allperm_ax2, ax2], ignore_index=True)
    #                     else:
    #                         allperm_ax2=ax2
                            
                        
    #                     msg = "CAP " + str(capn) + " perm " + str(perm) + ": ax1 = " + str(allperm_ax1.shape) + ", ax2 = " + str(allperm_ax2.shape)
    #                     logging.info(msg)                
                        
                        
    #                     del ax1, ax2
                        

                        
                
    #                 ##----------------------------------------------------------------
    #                 #  Display for each CAP
    #                 ##----------------------------------------------------------------

    #                 data=pd.concat([allperm_ax1,allperm_ax2], axis=1)
    #                 data.columns.values[0] = "wcbsv_dur"
    #                 data.columns.values[1] = "wcwsv_dwt"
    #                 msg=str(data)
    #                 logging.info(msg)
                    
    #                 subgrouppal=sns.color_palette("viridis_r", as_cmap=True)
    #                 # sns.kdeplot(x=data.wcbsv_dur, y=data.wcwsv_dwt, hue=cThcol[0], palette=subgrouppal, fill=True)
    #                 g = sns.jointplot(data=data, x="wcbsv_dur", y="wcwsv_dwt", hue=cThcol[0], s=10, palette=subgrouppal, alpha=.1, height=8, space=0.2, ratio=2, xlim = (0,0.05), ylim = (0,10))
    #                 # g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
    #                 # g.ax_joint.collections[0].set_alpha(0)
    #                 g.set_axis_labels("$BSV(FO)$", "$WSV(DT)$");
    #                 g.ax_joint.legend_.remove()
                    
                    
    #             # plt.show()
    #             savefilen = filein.outdir + 'P' + str(pth) + '_2dScatter_BSV_FO_WSV_DT_cap' + str(capn) + '_' + cThcol[0] + '.png'
    #             plt.savefig(savefilen, bbox_inches='tight')
    #             plt.close()
                
    #             del allperm_ax1, allperm_ax2




# - Notify job completion
msg = "\n========== The jobs are all completed. ==========\n"
logging.info(msg)











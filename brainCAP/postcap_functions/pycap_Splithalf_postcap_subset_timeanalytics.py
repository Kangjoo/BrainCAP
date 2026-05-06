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
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA #Principal Component Analysis


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
                    filename=filein.logdir + 'pycap_Splithalf_postcap_subset_timeanalytics.log',
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

if param.seedtype == "seedbased":
    pthrange = param.sig_threshold_range
elif param.seedtype == "seedfree":
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

    if param.seedtype == "seedbased":
        param.sig_threshold = pth
    elif param.seedtype == "seedfree":
        param.randTthreshold = pth
        
        
    # Define variables
    colcol = ['0','1','2','3','4'] 
    mypal=sns.color_palette("viridis_r", as_cmap=True)
    mypal2=[[0,0,0],[0,0,0],[0,0,0]]    
    mypal3=[[0.5,0.5,0.5],[1,1,1]]
    mypal4=["#FC6C85","#ABC5FE","#A1785C"]
    palette_Paired = sns.color_palette("Paired", 10)
    mypal_5cap = sns.color_palette([palette_Paired[5], palette_Paired[1], palette_Paired[3],palette_Paired[9],palette_Paired[7]])

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
    # #        Part I:   Individual Mean Data -both days
    # # 
    # # -------------------------------------------------------------


    # ---------- Individual distribution for DwellTime_WSmean in 3 states ------------#
    dataname="DwellTime_WSmean"
    df_mDT = pd.DataFrame({'subject': np.arange(1, len(filein.sublist)+1), 'subID': filein.sublist})
    
    capdata_0=capdwt_wsm_allcap_r_allday_allperm[["subID", "0", "half","permidx", "day"]]
    df_mDT['capdata_0_mDT']=convert_capdata_indscore(capdata_0,filein,"0")
    
    capdata_1=capdwt_wsm_allcap_r_allday_allperm[["subID", "1", "half","permidx", "day"]]
    df_mDT['capdata_1_mDT']=convert_capdata_indscore(capdata_1,filein,"1")
    
    capdata_2=capdwt_wsm_allcap_r_allday_allperm[["subID", "2", "half","permidx", "day"]]
    df_mDT['capdata_2_mDT']=convert_capdata_indscore(capdata_2,filein,"2")
    
    capdata_3=capdwt_wsm_allcap_r_allday_allperm[["subID", "3", "half","permidx", "day"]]
    df_mDT['capdata_3_mDT']=convert_capdata_indscore(capdata_3,filein,"3")
    
    capdata_4=capdwt_wsm_allcap_r_allday_allperm[["subID", "4", "half","permidx", "day"]]
    df_mDT['capdata_4_mDT']=convert_capdata_indscore(capdata_4,filein,"4")
    
    
    df_mDT["state1_mDT"]=df_mDT[['capdata_0_mDT', 'capdata_1_mDT']].mean(axis=1)
    df_mDT["state2_mDT"]=df_mDT[['capdata_1_mDT', 'capdata_2_mDT']].mean(axis=1)
    df_mDT["state3_mDT"]=df_mDT['capdata_4_mDT']
        
    df_mDT.to_csv(filein.outdir + 'P' + str(pth) + '.0_' + dataname + '_subgroup_allcTh.csv')
    del capdata_0, capdata_1, capdata_2, capdata_3, capdata_4


    # ---------- Individual distribution for DwellTime_WSstd in 3 states ------------#
    dataname="DwellTime_WSstd"
    df_vDT = pd.DataFrame({'subject': np.arange(1, len(filein.sublist)+1), 'subID': filein.sublist})
    
    capdata_0=capdwt_wsv_allcap_r_allday_allperm[["subID", "0", "half","permidx", "day"]]
    df_vDT['capdata_0_vDT']=convert_capdata_indscore(capdata_0,filein,"0")
    
    capdata_1=capdwt_wsv_allcap_r_allday_allperm[["subID", "1", "half","permidx", "day"]]
    df_vDT['capdata_1_vDT']=convert_capdata_indscore(capdata_1,filein,"1")
    
    capdata_2=capdwt_wsv_allcap_r_allday_allperm[["subID", "2", "half","permidx", "day"]]
    df_vDT['capdata_2_vDT']=convert_capdata_indscore(capdata_2,filein,"2")
    
    capdata_3=capdwt_wsv_allcap_r_allday_allperm[["subID", "3", "half","permidx", "day"]]
    df_vDT['capdata_3_vDT']=convert_capdata_indscore(capdata_3,filein,"3")
    
    capdata_4=capdwt_wsv_allcap_r_allday_allperm[["subID", "4", "half","permidx", "day"]]
    df_vDT['capdata_4_vDT']=convert_capdata_indscore(capdata_4,filein,"4")
    
    
    df_vDT["state1_vDT"]=df_vDT[['capdata_0_vDT', 'capdata_1_vDT']].mean(axis=1)
    df_vDT["state2_vDT"]=df_vDT[['capdata_1_vDT', 'capdata_2_vDT']].mean(axis=1)
    df_vDT["state3_vDT"]=df_vDT['capdata_4_vDT']
        
    df_vDT.to_csv(filein.outdir + 'P' + str(pth) + '.0_' + dataname + '_subgroup_allcTh.csv')
    del capdata_0, capdata_1, capdata_2, capdata_3, capdata_4





    # ---------- Individual distribution for Occupancy_individual in 3 states ------------#
    dataname="Occupancy_individual"
    df_FO = pd.DataFrame({'subject': np.arange(1, len(filein.sublist)+1), 'subID': filein.sublist})
    
    capdata_0=capdur_ind_allcap_r_allday_allperm[["subID", "0", "half","permidx", "day"]]
    df_FO['capdata_0_FO']=convert_capdata_indscore(capdata_0,filein,"0")
    
    capdata_1=capdur_ind_allcap_r_allday_allperm[["subID", "1", "half","permidx", "day"]]
    df_FO['capdata_1_FO']=convert_capdata_indscore(capdata_1,filein,"1")
    
    capdata_2=capdur_ind_allcap_r_allday_allperm[["subID", "2", "half","permidx", "day"]]
    df_FO['capdata_2_FO']=convert_capdata_indscore(capdata_2,filein,"2")
    
    capdata_3=capdur_ind_allcap_r_allday_allperm[["subID", "3", "half","permidx", "day"]]
    df_FO['capdata_3_FO']=convert_capdata_indscore(capdata_3,filein,"3")
    
    capdata_4=capdur_ind_allcap_r_allday_allperm[["subID", "4", "half","permidx", "day"]]
    df_FO['capdata_4_FO']=convert_capdata_indscore(capdata_4,filein,"4")
    
    
    df_FO["state1_FO"]=df_FO[['capdata_0_FO', 'capdata_1_FO']].mean(axis=1)
    df_FO["state2_FO"]=df_FO[['capdata_1_FO', 'capdata_2_FO']].mean(axis=1)
    df_FO["state3_FO"]=df_FO['capdata_4_FO']
        
    df_FO.to_csv(filein.outdir + 'P' + str(pth) + '.0_' + dataname + '_subgroup_allcTh.csv')
    del capdata_0, capdata_1, capdata_2, capdata_3, capdata_4





    # ---------- Combine 9 Neural features and estimate subgroups ------------#
    df_nf = pd.merge(df_FO, df_mDT, on = ["subject","subID"])
    df_nf = pd.merge(df_nf, df_vDT, on = ["subject","subID"])

    del df_FO, df_mDT, df_vDT


    # # -------------------------------------------------------------
    # # 
    # #         Part I:  Subgrouping Data - average over days
    # # 
    # # -------------------------------------------------------------



    # ----------- K-means clustering of subjects using 9 neural features ------------ #
    X=df_nf[["state1_FO","state2_FO","state3_FO","state1_mDT","state2_mDT","state3_mDT","state1_vDT","state2_vDT","state3_vDT"]]
    
    msg = str(X)
    logging.info(msg)    
    X=(X-X.mean())/X.std()
    import scipy.cluster.hierarchy as sch
    plt.figure(figsize=(10, 7))  
    plt.title("Dendrograms")  
    dend = sch.dendrogram(sch.linkage(X, method='ward'))
    plt.savefig(filein.outdir + 'P' + str(pth) + '.0_FO_mDT_vDT_subgroup_9nf_dendrogram.png')
    plt.close()
    
    
    
    clustering = AgglomerativeClustering(n_clusters=3, linkage='ward').fit(X)
    msg = str(clustering.labels_)
    logging.info(msg)
    df_nf["subgroup_9nf"]=clustering.labels_ + 1
    df_nf.to_csv(filein.outdir + 'P' + str(pth) + '.0_FO_mDT_vDT_subgroup_9nf.csv')

    df_s1 = df_nf[df_nf['subgroup_9nf'] == 1]
    df_s1 = df_s1['subID']
    df_s2 = df_nf[df_nf['subgroup_9nf'] == 2]
    df_s2 = df_s2['subID']
    df_s3 = df_nf[df_nf['subgroup_9nf'] == 3]
    df_s3 = df_s3['subID']

    df_s1.to_csv(filein.outdir + 'subgroup1_9nf.csv',index=False)
    df_s2.to_csv(filein.outdir + 'subgroup2_9nf.csv',index=False)
    df_s3.to_csv(filein.outdir + 'subgroup3_9nf.csv',index=False)

    
    
    # ------------ PCA on cluster labels for 2d embedding ------------- #
    
    pca_all = PCA()
    PCs_all = pd.DataFrame(pca_all.fit_transform(X))
    PCs_all.columns = ["neuralPC1", "neuralPC2", "neuralPC3", "neuralPC4", "neuralPC5", "neuralPC6", "neuralPC7", "neuralPC8", "neuralPC9"]
    PCs_all["subgroup_9nf"]=df_nf["subgroup_9nf"]
    PCs_all["subID"]=df_nf["subID"]
    msg = str(PCs_all)
    logging.info(msg)
        
    sns.scatterplot(data=PCs_all, x="neuralPC1", y="neuralPC2", hue="subgroup_9nf", palette=mypal, s=70,edgecolor="black")
    plt.savefig(filein.outdir + 'P' + str(pth) + '.0_FO_mDT_vDT_subgroup_PC12.png')
    plt.close()
    
    PCs_all.to_csv(filein.outdir + 'P' + str(pth) + '.0_FO_mDT_vDT_neuralPCs_9nf.csv')

    # ------------ PCA on cluster labels for 2d embedding ------------- #

    pca_all.fit(X)
    PC_values = np.arange(pca_all.n_components_) + 1
    plt.plot(PC_values, pca_all.explained_variance_ratio_, 'o-', linewidth=2, color='black',markersize=15)
    plt.ylim((0, 1))
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')
    plt.savefig(filein.outdir + 'P' + str(pth) + '.0_FO_mDT_vDT_subgroup_9nf_PCA_screeplot.png')
    msg="Explained variance(%) : " + str(pca_all.explained_variance_ratio_)
    logging.info(msg)
    plt.close()
    
    loadings = pca_all.components_
    num_pc = pca_all.n_features_
    pc_list = ["PC"+str(i) for i in list(range(1, num_pc+1))]
    loadings_df = pd.DataFrame.from_dict(dict(zip(pc_list, loadings)))
    loadings_df['variable'] =X.columns.values
    loadings_df['capn'] = [1,2,3,1,2,3,1,2,3]
    #loadings_df = loadings_df.set_index('variable')
    msg = str(loadings_df)
    logging.info(msg)
    
    sns.barplot(data=loadings_df, x="variable", y="PC1", hue="capn", palette=mypal4, edgecolor="black", dodge=False)
    plt.ylim((0,0.7))
    plt.savefig(filein.outdir + 'P' + str(pth) + '.0_FO_mDT_vDT_subgroup_9nf_PC1_loading_barplot.png')
    plt.close()

    sns.barplot(data=loadings_df, x="variable", y="PC2", hue="capn", palette=mypal4, edgecolor="black", dodge=False)
    plt.ylim((0,0.7))
    plt.savefig(filein.outdir + 'P' + str(pth) + '.0_FO_mDT_vDT_subgroup_9nf_PC2_loading_barplot.png')
    plt.close()
    
    sns.barplot(data=loadings_df, x="variable", y="PC3", hue="capn", palette=mypal4, edgecolor="black", dodge=False)
    plt.ylim((0,0.7))
    plt.savefig(filein.outdir + 'P' + str(pth) + '.0_FO_mDT_vDT_subgroup_9nf_PC3_loading_barplot.png')
    plt.close()
    

    del df_nf, df_s1, df_s2, df_s3, X, clustering, pca_all, PCs_all, PC_values, num_pc, pc_list, loadings_df



    # # -------------------------------------------------------------
    # # 
    # #         Part II: Individual Mean Data - day 1
    # # 
    # # -------------------------------------------------------------


    # ---------- Individual distribution for DwellTime_WSmean in 3 states ------------#
    dataname="DwellTime_WSmean"
    df_mDT_day1 = pd.DataFrame({'subject': np.arange(1, len(filein.sublist)+1), 'subID': filein.sublist})
    
    capdata_0=capdwt_wsm_allcap_r_allday_allperm[["subID", "0", "half","permidx", "day"]]
    capdata_0_day1=capdata_0[capdata_0["day"] == 1]
    df_mDT_day1['capdata_0_mDT_day1']=convert_capdata_indscore(capdata_0_day1,filein,"0")
    
    capdata_1=capdwt_wsm_allcap_r_allday_allperm[["subID", "1", "half","permidx", "day"]]
    capdata_1_day1=capdata_1[capdata_1["day"] == 1]
    df_mDT_day1['capdata_1_mDT_day1']=convert_capdata_indscore(capdata_1_day1,filein,"1")
    
    capdata_2=capdwt_wsm_allcap_r_allday_allperm[["subID", "2", "half","permidx", "day"]]
    capdata_2_day1=capdata_2[capdata_2["day"] == 1]
    df_mDT_day1['capdata_2_mDT_day1']=convert_capdata_indscore(capdata_2_day1,filein,"2")
    
    capdata_3=capdwt_wsm_allcap_r_allday_allperm[["subID", "3", "half","permidx", "day"]]
    capdata_3_day1=capdata_3[capdata_3["day"] == 1]
    df_mDT_day1['capdata_3_mDT_day1']=convert_capdata_indscore(capdata_3_day1,filein,"3")
    
    capdata_4=capdwt_wsm_allcap_r_allday_allperm[["subID", "4", "half","permidx", "day"]]
    capdata_4_day1=capdata_4[capdata_4["day"] == 1]
    df_mDT_day1['capdata_4_mDT_day1']=convert_capdata_indscore(capdata_4_day1,filein,"4")
    
    
    df_mDT_day1["state1_mDT_day1"]=df_mDT_day1[['capdata_0_mDT_day1', 'capdata_1_mDT_day1']].mean(axis=1)
    df_mDT_day1["state2_mDT_day1"]=df_mDT_day1[['capdata_1_mDT_day1', 'capdata_2_mDT_day1']].mean(axis=1)
    df_mDT_day1["state3_mDT_day1"]=df_mDT_day1['capdata_4_mDT_day1']
        
    del capdata_0, capdata_1, capdata_2, capdata_3, capdata_4
    
    

    # ---------- Individual distribution for DwellTime_WSstd in 3 states ------------#
    dataname="DwellTime_WSstd"
    df_vDT_day1 = pd.DataFrame({'subject': np.arange(1, len(filein.sublist)+1), 'subID': filein.sublist})
    
    capdata_0=capdwt_wsv_allcap_r_allday_allperm[["subID", "0", "half","permidx", "day"]]
    capdata_0_day1=capdata_0[capdata_0["day"] == 1]
    df_vDT_day1['capdata_0_vDT_day1']=convert_capdata_indscore(capdata_0_day1,filein,"0")
    
    capdata_1=capdwt_wsv_allcap_r_allday_allperm[["subID", "1", "half","permidx", "day"]]
    capdata_1_day1=capdata_1[capdata_1["day"] == 1]
    df_vDT_day1['capdata_1_vDT_day1']=convert_capdata_indscore(capdata_1_day1,filein,"1")
    
    capdata_2=capdwt_wsv_allcap_r_allday_allperm[["subID", "2", "half","permidx", "day"]]
    capdata_2_day1=capdata_2[capdata_2["day"] == 1]
    df_vDT_day1['capdata_2_vDT_day1']=convert_capdata_indscore(capdata_2_day1,filein,"2")
    
    capdata_3=capdwt_wsv_allcap_r_allday_allperm[["subID", "3", "half","permidx", "day"]]
    capdata_3_day1=capdata_3[capdata_3["day"] == 1]
    df_vDT_day1['capdata_3_vDT_day1']=convert_capdata_indscore(capdata_3_day1,filein,"3")
    
    capdata_4=capdwt_wsv_allcap_r_allday_allperm[["subID", "4", "half","permidx", "day"]]
    capdata_4_day1=capdata_4[capdata_4["day"] == 1]
    df_vDT_day1['capdata_4_vDT_day1']=convert_capdata_indscore(capdata_4_day1,filein,"4")
    
    
    df_vDT_day1["state1_vDT_day1"]=df_vDT_day1[['capdata_0_vDT_day1', 'capdata_1_vDT_day1']].mean(axis=1)
    df_vDT_day1["state2_vDT_day1"]=df_vDT_day1[['capdata_1_vDT_day1', 'capdata_2_vDT_day1']].mean(axis=1)
    df_vDT_day1["state3_vDT_day1"]=df_vDT_day1['capdata_4_vDT_day1']
        
    del capdata_0, capdata_1, capdata_2, capdata_3, capdata_4
    
    

    # ---------- Individual distribution for Occupancy_individual in 3 states ------------#
    dataname="Occupancy_individual"
    df_FO_day1 = pd.DataFrame({'subject': np.arange(1, len(filein.sublist)+1), 'subID': filein.sublist})
    
    capdata_0=capdur_ind_allcap_r_allday_allperm[["subID", "0", "half","permidx", "day"]]
    capdata_0_day1=capdata_0[capdata_0["day"] == 1]
    df_FO_day1['capdata_0_FO_day1']=convert_capdata_indscore(capdata_0_day1,filein,"0")
    
    capdata_1=capdur_ind_allcap_r_allday_allperm[["subID", "1", "half","permidx", "day"]]
    capdata_1_day1=capdata_1[capdata_1["day"] == 1]
    df_FO_day1['capdata_1_FO_day1']=convert_capdata_indscore(capdata_1_day1,filein,"1")
    
    capdata_2=capdur_ind_allcap_r_allday_allperm[["subID", "2", "half","permidx", "day"]]
    capdata_2_day1=capdata_2[capdata_2["day"] == 1]
    df_FO_day1['capdata_2_FO_day1']=convert_capdata_indscore(capdata_2_day1,filein,"2")
    
    capdata_3=capdur_ind_allcap_r_allday_allperm[["subID", "3", "half","permidx", "day"]]
    capdata_3_day1=capdata_3[capdata_3["day"] == 1]
    df_FO_day1['capdata_3_FO_day1']=convert_capdata_indscore(capdata_3_day1,filein,"3")
    
    capdata_4=capdur_ind_allcap_r_allday_allperm[["subID", "4", "half","permidx", "day"]]
    capdata_4_day1=capdata_4[capdata_4["day"] == 1]
    df_FO_day1['capdata_4_FO_day1']=convert_capdata_indscore(capdata_4_day1,filein,"4")
    
    
    df_FO_day1["state1_FO_day1"]=df_FO_day1[['capdata_0_FO_day1', 'capdata_1_FO_day1']].mean(axis=1)
    df_FO_day1["state2_FO_day1"]=df_FO_day1[['capdata_1_FO_day1', 'capdata_2_FO_day1']].mean(axis=1)
    df_FO_day1["state3_FO_day1"]=df_FO_day1['capdata_4_FO_day1']
        
    del capdata_0, capdata_1, capdata_2, capdata_3, capdata_4
    
    

    # ---------- Combine 9 Neural features and estimate subgroups ------------#
    df_nf_day1 = pd.merge(df_FO_day1, df_mDT_day1, on = ["subject","subID"])
    df_nf_day1 = pd.merge(df_nf_day1, df_vDT_day1, on = ["subject","subID"])

    df_nf_day1.to_csv(filein.outdir + 'P' + str(pth) + '.0_FO_mDT_vDT_day1_allcTh.csv')

    del df_FO_day1, df_mDT_day1, df_vDT_day1

    # # -------------------------------------------------------------
    # # 
    # #          Part II: Individual Mean Data - day 2
    # # 
    # # -------------------------------------------------------------


    # ---------- Individual distribution for DwellTime_WSmean in 3 states ------------#
    dataname="DwellTime_WSmean"
    df_mDT_day2 = pd.DataFrame({'subject': np.arange(1, len(filein.sublist)+1), 'subID': filein.sublist})
    
    capdata_0=capdwt_wsm_allcap_r_allday_allperm[["subID", "0", "half","permidx", "day"]]
    capdata_0_day2=capdata_0[capdata_0["day"] == 2]
    df_mDT_day2['capdata_0_mDT_day2']=convert_capdata_indscore(capdata_0_day2,filein,"0")
    
    capdata_1=capdwt_wsm_allcap_r_allday_allperm[["subID", "1", "half","permidx", "day"]]
    capdata_1_day2=capdata_1[capdata_1["day"] == 2]
    df_mDT_day2['capdata_1_mDT_day2']=convert_capdata_indscore(capdata_1_day2,filein,"1")
    
    capdata_2=capdwt_wsm_allcap_r_allday_allperm[["subID", "2", "half","permidx", "day"]]
    capdata_2_day2=capdata_2[capdata_2["day"] == 2]
    df_mDT_day2['capdata_2_mDT_day2']=convert_capdata_indscore(capdata_2_day2,filein,"2")
    
    capdata_3=capdwt_wsm_allcap_r_allday_allperm[["subID", "3", "half","permidx", "day"]]
    capdata_3_day2=capdata_3[capdata_3["day"] == 2]
    df_mDT_day2['capdata_3_mDT_day2']=convert_capdata_indscore(capdata_3_day2,filein,"3")
    
    capdata_4=capdwt_wsm_allcap_r_allday_allperm[["subID", "4", "half","permidx", "day"]]
    capdata_4_day2=capdata_4[capdata_4["day"] == 2]
    df_mDT_day2['capdata_4_mDT_day2']=convert_capdata_indscore(capdata_4_day2,filein,"4")
    
    
    df_mDT_day2["state1_mDT_day2"]=df_mDT_day2[['capdata_0_mDT_day2', 'capdata_1_mDT_day2']].mean(axis=1)
    df_mDT_day2["state2_mDT_day2"]=df_mDT_day2[['capdata_1_mDT_day2', 'capdata_2_mDT_day2']].mean(axis=1)
    df_mDT_day2["state3_mDT_day2"]=df_mDT_day2['capdata_4_mDT_day2']
        
    del capdata_0, capdata_1, capdata_2, capdata_3, capdata_4
    
    

    # ---------- Individual distribution for DwellTime_WSstd in 3 states ------------#
    dataname="DwellTime_WSstd"
    df_vDT_day2 = pd.DataFrame({'subject': np.arange(1, len(filein.sublist)+1), 'subID': filein.sublist})
    
    capdata_0=capdwt_wsv_allcap_r_allday_allperm[["subID", "0", "half","permidx", "day"]]
    capdata_0_day2=capdata_0[capdata_0["day"] == 2]
    df_vDT_day2['capdata_0_vDT_day2']=convert_capdata_indscore(capdata_0_day2,filein,"0")
    
    capdata_1=capdwt_wsv_allcap_r_allday_allperm[["subID", "1", "half","permidx", "day"]]
    capdata_1_day2=capdata_1[capdata_1["day"] == 2]
    df_vDT_day2['capdata_1_vDT_day2']=convert_capdata_indscore(capdata_1_day2,filein,"1")
    
    capdata_2=capdwt_wsv_allcap_r_allday_allperm[["subID", "2", "half","permidx", "day"]]
    capdata_2_day2=capdata_2[capdata_2["day"] == 2]
    df_vDT_day2['capdata_2_vDT_day2']=convert_capdata_indscore(capdata_2_day2,filein,"2")
    
    capdata_3=capdwt_wsv_allcap_r_allday_allperm[["subID", "3", "half","permidx", "day"]]
    capdata_3_day2=capdata_3[capdata_3["day"] == 2]
    df_vDT_day2['capdata_3_vDT_day2']=convert_capdata_indscore(capdata_3_day2,filein,"3")
    
    capdata_4=capdwt_wsv_allcap_r_allday_allperm[["subID", "4", "half","permidx", "day"]]
    capdata_4_day2=capdata_4[capdata_4["day"] == 2]
    df_vDT_day2['capdata_4_vDT_day2']=convert_capdata_indscore(capdata_4_day2,filein,"4")
    
    
    df_vDT_day2["state1_vDT_day2"]=df_vDT_day2[['capdata_0_vDT_day2', 'capdata_1_vDT_day2']].mean(axis=1)
    df_vDT_day2["state2_vDT_day2"]=df_vDT_day2[['capdata_1_vDT_day2', 'capdata_2_vDT_day2']].mean(axis=1)
    df_vDT_day2["state3_vDT_day2"]=df_vDT_day2['capdata_4_vDT_day2']
        
    del capdata_0, capdata_1, capdata_2, capdata_3, capdata_4
    
    

    # ---------- Individual distribution for Occupancy_individual in 3 states ------------#
    dataname="Occupancy_individual"
    df_FO_day2 = pd.DataFrame({'subject': np.arange(1, len(filein.sublist)+1), 'subID': filein.sublist})
    
    capdata_0=capdur_ind_allcap_r_allday_allperm[["subID", "0", "half","permidx", "day"]]
    capdata_0_day2=capdata_0[capdata_0["day"] == 2]
    df_FO_day2['capdata_0_FO_day2']=convert_capdata_indscore(capdata_0_day2,filein,"0")
    
    capdata_1=capdur_ind_allcap_r_allday_allperm[["subID", "1", "half","permidx", "day"]]
    capdata_1_day2=capdata_1[capdata_1["day"] == 2]
    df_FO_day2['capdata_1_FO_day2']=convert_capdata_indscore(capdata_1_day2,filein,"1")
    
    capdata_2=capdur_ind_allcap_r_allday_allperm[["subID", "2", "half","permidx", "day"]]
    capdata_2_day2=capdata_2[capdata_2["day"] == 2]
    df_FO_day2['capdata_2_FO_day2']=convert_capdata_indscore(capdata_2_day2,filein,"2")
    
    capdata_3=capdur_ind_allcap_r_allday_allperm[["subID", "3", "half","permidx", "day"]]
    capdata_3_day2=capdata_3[capdata_3["day"] == 2]
    df_FO_day2['capdata_3_FO_day2']=convert_capdata_indscore(capdata_3_day2,filein,"3")
    
    capdata_4=capdur_ind_allcap_r_allday_allperm[["subID", "4", "half","permidx", "day"]]
    capdata_4_day2=capdata_4[capdata_4["day"] == 2]
    df_FO_day2['capdata_4_FO_day2']=convert_capdata_indscore(capdata_4_day2,filein,"4")
    
    
    df_FO_day2["state1_FO_day2"]=df_FO_day2[['capdata_0_FO_day2', 'capdata_1_FO_day2']].mean(axis=1)
    df_FO_day2["state2_FO_day2"]=df_FO_day2[['capdata_1_FO_day2', 'capdata_2_FO_day2']].mean(axis=1)
    df_FO_day2["state3_FO_day2"]=df_FO_day2['capdata_4_FO_day2']
        
    del capdata_0, capdata_1, capdata_2, capdata_3, capdata_4
    
    

    # ---------- Combine 9 Neural features and estimate subgroups ------------#
    df_nf_day2 = pd.merge(df_FO_day2, df_mDT_day2, on = ["subject","subID"])
    df_nf_day2 = pd.merge(df_nf_day2, df_vDT_day2, on = ["subject","subID"])

    df_nf_day2.to_csv(filein.outdir + 'P' + str(pth) + '.0_FO_mDT_vDT_day2_allcTh.csv')


    del df_FO_day2, df_mDT_day2, df_vDT_day2
    
    


    # # -------------------------------------------------------------
    # # 
    # #         Part II: Subgrouping Data - using each day neural features separately for clustering
    # # 
    # # -------------------------------------------------------------

    
    df_nf = pd.merge(df_nf_day1, df_nf_day2, on = ["subject","subID"])
        
    
    del df_nf_day1, df_nf_day2
    
    # - Clustering
    from sklearn.preprocessing import normalize
    X=df_nf[["capdata_0_FO_day1","capdata_1_FO_day1","capdata_2_FO_day1","capdata_3_FO_day1","capdata_4_FO_day1",
                "capdata_0_mDT_day1","capdata_1_mDT_day1","capdata_2_mDT_day1","capdata_3_mDT_day1","capdata_4_mDT_day1",
                "capdata_0_vDT_day1","capdata_1_vDT_day1","capdata_2_vDT_day1","capdata_3_vDT_day1","capdata_4_vDT_day1",
                "capdata_0_FO_day2","capdata_1_FO_day2","capdata_2_FO_day2","capdata_3_FO_day2","capdata_4_FO_day2",
                "capdata_0_mDT_day2","capdata_1_mDT_day2","capdata_2_mDT_day2","capdata_3_mDT_day2","capdata_4_mDT_day2",
                "capdata_0_vDT_day2","capdata_1_vDT_day2","capdata_2_vDT_day2","capdata_3_vDT_day2","capdata_4_vDT_day2"]]     
    msg = str(X)
    logging.info(msg)    
    X=(X-X.mean())/X.std()
    import scipy.cluster.hierarchy as sch
    plt.figure(figsize=(10, 7))  
    plt.title("Dendrograms")  
    sch.set_link_color_palette(["#FDE725FF","#440154FF","#22A884FF"])
    sch.dendrogram(sch.linkage(X, method='ward'))
    plt.savefig(filein.outdir + 'P' + str(pth) + '.0_FO_mDT_vDT_subgroup_30nf_dendrogram.png')
    plt.close()
    
    clustering = AgglomerativeClustering(n_clusters=3, linkage='ward').fit(X)
    msg = str(clustering.labels_)
    logging.info(msg)
    
    # - Save dataframe with clustering results
    df_nf["subgroup_30nf"]=clustering.labels_ + 1
    df_nf.to_csv(filein.outdir + 'P' + str(pth) + '.0_FO_mDT_vDT_subgroup_30nf.csv')

    df_s1 = df_nf[df_nf['subgroup_30nf'] == 1]
    df_s1 = df_s1['subID']
    df_s2 = df_nf[df_nf['subgroup_30nf'] == 2]
    df_s2 = df_s2['subID']
    df_s3 = df_nf[df_nf['subgroup_30nf'] == 3]
    df_s3 = df_s3['subID']
    
    df_s1.to_csv(filein.outdir + 'subgroup1_30nf.csv',index=False)
    df_s2.to_csv(filein.outdir + 'subgroup2_30nf.csv',index=False)
    df_s3.to_csv(filein.outdir + 'subgroup3_30nf.csv',index=False)


    del dend, clustering, df_s1, df_s2, df_s3
    
    # ------------ PCA on cluster labels for 2d embedding ------------- #
    
    pca_all = PCA()
    PCs_all = pd.DataFrame(pca_all.fit_transform(X))

    PCs_all.columns = ["neuralPC1", "neuralPC2", "neuralPC3", "neuralPC4", "neuralPC5", "neuralPC6", "neuralPC7", "neuralPC8", "neuralPC9","neuralPC10",
                        "neuralPC11","neuralPC12","neuralPC13","neuralPC14","neuralPC15", "neuralPC16", "neuralPC17", "neuralPC18", "neuralPC19","neuralPC20",
                        "neuralPC21","neuralPC22","neuralPC23","neuralPC24","neuralPC25", "neuralPC26", "neuralPC27", "neuralPC28", "neuralPC29","neuralPC30"]
    PCs_all["subgroup_30nf"]=df_nf["subgroup_30nf"]
    PCs_all["subID"]=df_nf["subID"]
    msg = str(PCs_all)
    logging.info(msg)
    
    sns.scatterplot(data=PCs_all, x="neuralPC1", y="neuralPC2", hue="subgroup_30nf", palette=mypal, s=50,edgecolor="black")
    plt.xlim((-12,12))
    plt.ylim((-12,12))
    plt.savefig(filein.outdir + 'P' + str(pth) + '.0_FO_mDT_vDT_subgroup_30nf_PC12.png')
    plt.close()
    
    PCs_all.to_csv(filein.outdir + 'P' + str(pth) + '.0_FO_mDT_vDT_neuralPCs_30nf.csv')

    # ------------ PCA on cluster labels for 2d embedding ------------- #

    pca_all.fit(X)
    PC_values = np.arange(pca_all.n_components_) + 1
    plt.plot(PC_values, pca_all.explained_variance_ratio_, 'o-', linewidth=2, color='black',markersize=10)
    plt.ylim((0, 0.4))
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')
    plt.savefig(filein.outdir + 'P' + str(pth) + '.0_FO_mDT_vDT_subgroup_30nf_PCA_screeplot.png')
    msg="Explained variance(%) : " + str(pca_all.explained_variance_ratio_)
    logging.info(msg)
    plt.close()
    
    loadings = pca_all.components_
    num_pc = pca_all.n_features_
    pc_list = ["PC"+str(i) for i in list(range(1, num_pc+1))]
    loadings_df = pd.DataFrame.from_dict(dict(zip(pc_list, loadings)))
    loadings_df['variable'] =X.columns.values
    loadings_df['capn'] = [1,2,3,4,5,
                            1,2,3,4,5,
                            1,2,3,4,5,
                            1,2,3,4,5,
                            1,2,3,4,5,
                            1,2,3,4,5,]
    # loadings_df = loadings_df.set_index('variable')
    msg = str(loadings_df)
    logging.info(msg)
    loadings_df.to_csv(filein.outdir + 'P' + str(pth) + '.0_FO_mDT_vDT_neuralPCs_30nf_loadings.csv')
    
    
    sns.barplot(data=loadings_df, x="variable", y="PC1", hue="capn", palette=mypal_5cap, edgecolor="black", dodge=False)
    # plt.ylim((0,0.7))
    plt.legend('', frameon=False)
    plt.savefig(filein.outdir + 'P' + str(pth) + '.0_FO_mDT_vDT_subgroup_30nf_PC1_loading_barplot.png')
    plt.close()

    sns.barplot(data=loadings_df, x="variable", y="PC2", hue="capn", palette=mypal_5cap, edgecolor="black", dodge=False)
    # plt.ylim((0,0.7))
    plt.legend('', frameon=False)
    plt.savefig(filein.outdir + 'P' + str(pth) + '.0_FO_mDT_vDT_subgroup_30nf_PC2_loading_barplot.png')
    plt.close()

    sns.barplot(data=loadings_df, x="variable", y="PC3", hue="capn", palette=mypal_5cap, edgecolor="black", dodge=False)
    # plt.ylim((0,0.7))
    plt.legend('', frameon=False)
    plt.savefig(filein.outdir + 'P' + str(pth) + '.0_FO_mDT_vDT_subgroup_30nf_PC3_loading_barplot.png')
    plt.close()





# - Notify job completion
msg = "\n========== The jobs are all completed. ==========\n"
logging.info(msg)











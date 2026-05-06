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
parser.add_argument("-pn", "--behavpcn", nargs='+', type=int, help="Number of behavioral PCs")  
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
parser.add_argument("-bt", "--behavtag", type=str, help="Behavior type")
parser.add_argument("-b", "--behavfilen", dest="behavfile", required=True,
                    help="Behavior filename", type=lambda f: open(f))  
parser.add_argument("-bp", "--behavpcafilen", dest="behavpcafilen", required=True,
                    help="Behavior PCA filename", type=lambda f: open(f))  
parser.add_argument("-np", "--neuralpcafilen", dest="neuralpcafilen", required=True,
                    help="Neural PCA filename", type=lambda f: open(f))                     
parser.add_argument("-u", "--unit", type=str, help="(p/d)")  # parcel or dense
parser.add_argument("-sgk", "--ksubgroupfilen", dest="k_subgroupfile", required=True,
                    help="k-solution subgroup filename", type=lambda f: open(f))
parser.add_argument("-sgn", "--neuralsubgroupfilen", dest="neural_subgroupfile", required=True,
                    help="neural subgroup filename", type=lambda f: open(f))                    
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
param.behavpcn = args.behavpcn[0]

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
filein.neuralpca_filen = args.neuralpcafilen.name

filein.behav_filen = args.behavfile.name
filein.behavpca_filen = args.behavpcafilen.name
param.behavtag = args.behavtag
filein.ksubgroup_filen = args.k_subgroupfile.name
filein.neuralsubgroup_filen = args.neural_subgroupfile.name


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
                    filename=filein.logdir + 'pycap_Splithalf_postcap_subset_ksolution_to_behavior.log',
                    filemode='w')
console = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)


# -------------------------------------------
#          Define functions
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


if param.seedtype == "seedbased":
    pthrange = param.sig_threshold_range
elif param.seedtype == "seedfree":
    pthrange = param.randTthresholdrange
    
    
for pth in pthrange:

    if param.seedtype == "seedbased":
        param.sig_threshold = pth
    elif param.seedtype == "seedfree":
        param.randTthreshold = pth


    for cTh in param.classTh:
    
        cThstring = '_' + str(cTh) + 'nf'
        cThcol = ['subgroup' + cThstring]


        # - Load all data    
        
        neural_pca = pd.read_csv(filein.neuralpca_filen, index_col=0)
        msg = str(neural_pca)
        logging.info(msg)
        
        behavior = pd.read_csv(filein.behav_filen, sep='\t')
        msg = str(behavior)
        logging.info(msg)
        
        behavior_pca = pd.read_csv(filein.behavpca_filen, sep='\t')
        msg = str(behavior_pca)
        logging.info(msg)
        
        ksubgroup = pd.read_csv(filein.ksubgroup_filen, index_col=0)
        msg = str(ksubgroup)
        logging.info(msg)
        
        neuralsubgroup = pd.read_csv(filein.neuralsubgroup_filen, index_col=0)
        msg = str(neuralsubgroup)
        logging.info(msg)  
            
            
        # - scatter plot 
        
        df = ksubgroup[["k5-k4_counts_mean12_z"]].copy()
        df["neuralPC1"] = neural_pca["neuralPC1"]
        df["neuralPC2"] = neural_pca["neuralPC2"]
        df["neuralPC3"] = neural_pca["neuralPC3"]
        df["bPC1"] = behavior_pca["PC1"]
        df["Group"] = neural_pca[cThcol[0]] 
        
        msg = str(df)
        logging.info(msg)
        mypal = ["#FDE725FF","#21908CFF","#440154FF"]
        
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)    
        slope, intercept, r, p, std_err = stats.linregress(df['k5-k4_counts_mean12_z'],df['neuralPC1'])
        sns.set_palette(mypal)
        sns.lmplot(x='k5-k4_counts_mean12_z', y='neuralPC1', hue='Group', data=df, fit_reg=False, legend=False, scatter_kws={'linewidths':1,'edgecolor':'k', 's': 70})
        ax = sns.regplot(x="k5-k4_counts_mean12_z", y="neuralPC1", data=df, scatter_kws={"zorder":-1},
        line_kws={'color': "black", 'linewidth':5, 'label':"y={0:.1f}x+{1:.1f}".format(slope,intercept)})
        # ax.legend()  
        plt.xlim(-3.5, 3.5) 
        plt.ylim(-12, 12) 
        plt.title('n=337 (r=' + str(r) + ', p=' + str(p) + ')')
        savefilen = filein.outdir + 'P' + str(pth) + ".0_k5-4_subcount_neuralPC1_scatter.png"
        plt.savefig(savefilen, bbox_inches='tight')
        plt.close()        
        
    
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)    
        slope, intercept, r, p, std_err = stats.linregress(df['k5-k4_counts_mean12_z'],df['neuralPC2'])
        sns.set_palette(mypal)
        sns.lmplot(x='k5-k4_counts_mean12_z', y='neuralPC2', hue='Group', data=df, fit_reg=False, legend=False, scatter_kws={'linewidths':1,'edgecolor':'k', 's': 70})
        ax = sns.regplot(x="k5-k4_counts_mean12_z", y="neuralPC2", data=df, scatter_kws={"zorder":-1},
        line_kws={'color': "black", 'linewidth':5, 'label':"y={0:.1f}x+{1:.1f}".format(slope,intercept)})
        # ax.legend() 
        plt.xlim(-3.5, 3.5) 
        plt.ylim(-12, 12)
        plt.title('n=337 (r=' + str(r) + ', p=' + str(p) + ')')
        savefilen = filein.outdir + 'P' + str(pth) + ".0_k5-4_subcount_neuralPC2_scatter.png"
        plt.savefig(savefilen, bbox_inches='tight')
        plt.close()   
        
        
    
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)    
        slope, intercept, r, p, std_err = stats.linregress(df['k5-k4_counts_mean12_z'],df['neuralPC3'])
        sns.set_palette(mypal)
        sns.lmplot(x='k5-k4_counts_mean12_z', y='neuralPC3', hue='Group', data=df, fit_reg=False, legend=False, scatter_kws={'linewidths':1,'edgecolor':'k', 's': 70})
        ax = sns.regplot(x="k5-k4_counts_mean12_z", y="neuralPC3", data=df, scatter_kws={"zorder":-1},
        line_kws={'color': "black", 'linewidth':5, 'label':"y={0:.1f}x+{1:.1f}".format(slope,intercept)})
        # ax.legend()  
        plt.xlim(-3.5, 3.5) 
        plt.ylim(-10, 10)
        plt.title('n=337 (r=' + str(r) + ', p=' + str(p) + ')')
        savefilen = filein.outdir + 'P' + str(pth) + ".0_k5-4_subcount_neuralPC3_scatter.png"
        plt.savefig(savefilen, bbox_inches='tight')
        plt.close()   
        
    
    
    
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)    
        slope, intercept, r, p, std_err = stats.linregress(df['k5-k4_counts_mean12_z'],df['bPC1'])
        sns.set_palette(mypal)
        sns.lmplot(x='k5-k4_counts_mean12_z', y='bPC1', hue='Group', data=df, fit_reg=False, legend=False, scatter_kws={'linewidths':1,'edgecolor':'k', 's': 70})
        ax = sns.regplot(x="k5-k4_counts_mean12_z", y="bPC1", data=df, scatter_kws={"zorder":-1},
        line_kws={'color': "black", 'linewidth':5, 'label':"y={0:.1f}x+{1:.1f}".format(slope,intercept)})
        # ax.legend()   
        plt.xlim(-3.5, 3.5) 
        plt.ylim(-30, 17)    
        plt.title('n=337 (r=' + str(r) + ', p=' + str(p) + ')')
        savefilen = filein.outdir + 'P' + str(pth) + ".0_k5-4_subcount_behavioralPC1_scatter.png"
        plt.savefig(savefilen, bbox_inches='tight')
        plt.close()        
        
    
    
    
    
        # group comparisons of ksolution preference
    
        # select data for T test between subgroups
        g1 = df[(df["Group"] == 1)] 
        g2 = df[(df["Group"] == 2)] 
        g3 = df[(df["Group"] == 3)] 
        
        msg=str(g1)
        logging.info(msg)
        msg=str(g2)
        logging.info(msg)
        msg=str(g3)
        logging.info(msg)  
        
        # print results in log file
        t12, p12 = stats.ttest_ind(a=g1['k5-k4_counts_mean12_z'], b=g2['k5-k4_counts_mean12_z'])
        t13, p13 = stats.ttest_ind(a=g1['k5-k4_counts_mean12_z'], b=g3['k5-k4_counts_mean12_z'])
        t23, p23 = stats.ttest_ind(a=g2['k5-k4_counts_mean12_z'], b=g3['k5-k4_counts_mean12_z'])
    
        msg="\n"
        logging.info(msg)
        # msg = "Performs the two-sided two-sample Kolmogorov-Smirnov test for goodness of fit between subgroups...\n"
        msg = "k5-k4_counts_mean12_z: Performs the two-sided two-sample T test between subgroups...\n"
        logging.info(msg)
        msg = cThstring + " -- 'k5-k4_counts_mean12_z' : group 1(n=" + str(len(g1)) + ") vs group 2(n=" + str(len(g2)) + "), T statistic=" + str(t12) + ", p=" + str(p12)
        logging.info(msg)
        msg = cThstring + " -- 'k5-k4_counts_mean12_z' : group 1(n=" + str(len(g1)) + ") vs group 3(n=" + str(len(g3)) + "), T statistic=" + str(t13) + ", p=" + str(p13)
        logging.info(msg)
        msg = cThstring + " -- 'k5-k4_counts_mean12_z' : group 2(n=" + str(len(g2)) + ") vs group 3(n=" + str(len(g3)) + "), T statistic=" + str(t23) + ", p=" + str(p23)
        logging.info(msg)  
        

        plt.figure(figsize=(7, 7))
        plt.title('ksolution dist - ' + cThcol[0])
        ax=sns.violinplot(data=df, x="Group", y="k5-k4_counts_mean12_z", palette=["#FDE725FF","#21908CFF","#440154FF"], inner=None, linewidth=0)
        plt.setp(ax.collections, alpha=.5)
        sns.boxplot(data=df, x="Group", y="k5-k4_counts_mean12_z", palette=["#FDE725FF","#21908CFF","#440154FF"], width=0.3, boxprops={'zorder': 2})
        plt.ylabel('k5-k4_counts_mean12_z')
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        savefilen = filein.outdir + 'P' + str(pth) + ".0_k5-4_subcount_" + cThcol[0] +  "_violinplot.png"
        if savefigs:
            plt.savefig(savefilen, bbox_inches='tight')
        plt.close()



        

# - Notify job completion
msg = "\n========== The jobs are all completed. ==========\n"
logging.info(msg)
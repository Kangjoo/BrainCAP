#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Kangjoo Lee (kangjoo.lee@yale.edu)
# Created Date: 04/05/2022
# Last Updated: 05/17/2022
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
import numpy as np
import argparse
import itertools
import pandas as pd
import seaborn as sns
import logging
import ptitprince as pt
import matplotlib.collections as clt
import matplotlib.pyplot as plt
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
parser.add_argument("-nc", "--ncapofinterest", nargs='+',
                    type=int, help="[QC] number of caps of interest")
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
param.ncapofinterest = args.ncapofinterest

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
                    filename=filein.logdir + 'output_postcap_step2.log',
                    filemode='w')
console = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)


# -------------------------------------------
#          Define classes.
# -------------------------------------------


# -------------------------------------------
#          Main analysis starts here.
# -------------------------------------------

param.splittype = "Ssplit"
permsplit = [param.splittype] * (param.maxperm-param.minperm+1)
permsplit = np.array(permsplit)

allpth_qc = pd.DataFrame()
bc_dur_allpth = pd.DataFrame()

if param.seedtype == "seedfree":
    pthrange = param.randTthresholdrange

for pth in pthrange:

    if param.seedtype == "seedfree":
        param.randTthreshold = pth

    msg = "=========== signal threshold P = " + str(pth) + "% ==========="
    logging.info(msg)

    # -------------------------------------------
    #            Setup input directory
    # -------------------------------------------

    if param.seedtype == "seedfree":
        filein.indir = args.outdir + param.gsr + "_" + param.seedIDname + \
            "/P" + str(param.randTthreshold) + ".0/"

    # --------------------------------------------------------
    # - Load QC info
    # --------------------------------------------------------

    df = pd.read_csv(filein.indir + 'QC_output.csv')
    msg = "Load QC data in: " + filein.indir + 'QC_output.csv'
    logging.info(msg)
    allpth_qc = pd.concat([allpth_qc, df], axis=0, ignore_index=True)
    del df

    
    allpth_qc.to_csv(filein.outdir + 'allpth_qc.csv')


# ----------------------------------------------------------
# - QC plot: n_caps
# ----------------------------------------------------------
pal = sns.color_palette()


for pth in pthrange:
    f, ax = plt.subplots(figsize=(7, 7))
    mypth = allpth_qc[(allpth_qc["seedthr"] == pth)]
    sns.histplot(data=mypth, x="n_cap",hue="half", bins=[3,4,5,6],stat="count", multiple="dodge",shrink=.9, discrete=True, edgecolor="none")
    plt.title("n_cap")
    plt.xlim((2, 8))
    # plt.show()
    savefilen = filein.outdir + "ncap"+ "Pth" + str(pth) + ".png"
    if savefigs:
        plt.savefig(savefilen, bbox_inches='tight')


# ----------------------------------------------------------
# - QC plot: n_caps counts for two n
# ----------------------------------------------------------
df_nc = pd.DataFrame(columns=['nc', 'half', 'counts'])
df_all = pd.DataFrame(columns=['nc', 'half', 'counts'])
inter_df_nc = pd.DataFrame(columns=['nc', 'counts'])
inter_df_all = pd.DataFrame(columns=['nc', 'counts'])
z = 0
for nc in param.ncapofinterest:
    myncap = allpth_qc[(allpth_qc["n_cap"] == nc)]
    myncap_training = allpth_qc[(allpth_qc["n_cap"] == nc) & (allpth_qc["half"] == "training")]
    myncap_test = allpth_qc[(allpth_qc["n_cap"] == nc) & (allpth_qc["half"] == "test")]
    myncap_all = pd.concat([myncap_training, myncap_test], ignore_index=True)
    myncap_all.to_csv(filein.outdir + 'ncap' + str(nc) + '_all.csv')

    int_df = myncap_training[myncap_training['nperm'].isin(myncap_test['nperm'])]
    int_df.to_csv(filein.outdir + 'ncap_'+str(nc)+'_intersection_df.csv')

    msg = "occurrence of k=" + str(nc) + "(training) = " + \
        str(len(myncap_training)) + ", (test) = " + \
        str(len(myncap_test)) + ", (intersection) = " + str(len(int_df))
    logging.info(msg)

    my_array = np.array([[nc, 'split1', len(myncap_training)], [
                        nc, 'split2', len(myncap_test)]], dtype=object)
    df_nc = pd.DataFrame(my_array, columns=['nc', 'half', 'counts'])
    df_all = pd.concat([df_all, df_nc], ignore_index=True)

    my_array2 = np.array([[nc, len(int_df)]], dtype=object)
    inter_df_nc = pd.DataFrame(my_array2, columns=['nc', 'counts'])
    inter_df_all = pd.concat([inter_df_all, inter_df_nc], ignore_index=True)

    z = z+1

df_all["counts"]=df_all["counts"]/param.maxperm * 100
df_all.rename(columns = {'counts':'occurrence'}, inplace = True)

inter_df_all["counts"]=inter_df_all["counts"]/param.maxperm * 100
inter_df_all.rename(columns = {'counts':'occurrence'}, inplace = True)

df_all.to_csv(filein.outdir + 'ncap_ocurrence_all.csv')
inter_df_all.to_csv(filein.outdir + 'ncap_inter_occurrence_all.csv')

plt.figure(figsize=(8, 8))
sns.barplot(x="nc", y="occurrence", hue="half", data=df_all)
plt.ylim((-1, 100))
# plt.ylim((-1, param.maxperm))
# plt.show()
savefilen = filein.outdir + "ncap_occurrence.png"
if savefigs:
    plt.savefig(savefilen, bbox_inches='tight')

palette = sns.color_palette("tab20", 20)
pal2 = sns.color_palette([palette[10], palette[10]])
# pal2 = [(0, 0, 0),(1, 1, 1)]

plt.figure(figsize=(8, 8))
sns.barplot(x="nc", y="occurrence", data=inter_df_all,palette=pal2)
plt.ylim((-1, 100))
# plt.ylim((-1, param.maxperm))
# plt.show()
savefilen = filein.outdir + "ncap_inter_occurrence.png"
if savefigs:
    plt.savefig(savefilen, bbox_inches='tight')


# ----------------------------------------------------------
# - QC plot: fdim
# ----------------------------------------------------------

#define samples
group1 = allpth_qc[(allpth_qc['half']=='training') & (allpth_qc["seedthr"] == pth)]
group2 = allpth_qc[(allpth_qc['half']=='test') & (allpth_qc["seedthr"] == pth)]

#perform independent two sample t-test
tval, pval=stats.ttest_rel(group1['fdim_clusterID'], group2['fdim_clusterID'])


f, ax = plt.subplots(figsize=(8, 8))
ax = pt.RainCloud(x="seedthr", y="fdim_clusterID", hue="half", data=allpth_qc,
                  palette=pal, bw=.2, width_viol=.5,
                  ax=ax, orient="v", alpha=.85, dodge=True)
plt.title("no. selected frames T(" + str(tval) + ", P(" + str(pval) +")")
# plt.xlim((-1, 5))
# plt.show()
savefilen = filein.outdir + "fdim_clusterID.png"
if savefigs:
    plt.savefig(savefilen, bbox_inches='tight')




# - Notify job completion
msg = "\n========== The jobs are all completed. ==========\n"
logging.info(msg)

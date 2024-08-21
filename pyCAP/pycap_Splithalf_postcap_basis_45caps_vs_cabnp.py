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
from scipy import spatial
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
parser.add_argument("-pi", "--parcelinfofilen", dest="parcelinfofilen", required=True,
                    help="parcelinfofilen", type=lambda f: open(f))                    
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
filein.parcelinfo_filen = args.parcelinfofilen

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
                    filename=filein.logdir + 'pycap_Splithalf_postcap_basis_45caps_vs_cabnp.log',
                    filemode='w')
console = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)


# -------------------------------------------
#          Define functions
# -------------------------------------------


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


def plot_donut(labels, savefilen):
    grp=np.ones(len(labels))
    pal = ["#0000FF","#6400FF","#00FFFF","#990099","#00FF00","#009B9B","#FFFF00","#FA3EFB","#FF0000","#B15928","#FF9D00","#417D00"]
    plt.pie(grp, colors=pal,counterclock=False, startangle=-255)
    my_circle=plt.Circle( (0,0), 0.9, color='white')
    p=plt.gcf()
    p.gca().add_artist(my_circle)
    plt.savefig(savefilen, bbox_inches='tight')
    plt.close()      
    
    return

def plot_polar_graph(labels, values, bc, savefilen):
    num_vars=len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='b', linewidth=4, fill=False)
    ax.fill(angles, values, color='b', alpha=0.1)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles), labels)
    
    ax.set_ylim(-0.6,0.6)
    ax.set_yticks([-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6])
    y_tick_labels = [-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6]
    ax.set_yticklabels(y_tick_labels,color='w')
    plt.xticks(color = 'w')
    
    ind = y_tick_labels.index(0)  # find index of value 0
    gridlines = ax.yaxis.get_gridlines()
    gridlines[ind].set_color("k")
    gridlines[ind].set_linewidth(2.5)
    
    plt.title("CAP " + str(bc))
    plt.savefig(savefilen, bbox_inches='tight')
    plt.close()    
    
    return

def cosine_similarity(a,b):
    from numpy import dot
    from numpy.linalg import norm
    cossim = np.inner(a, b) / (norm(a) * norm(b))
    return cossim
    

def compare_cap_vs_networks(basisCAPs,parcnet,outfilen):
    
    netlist=parcnet.network_id.unique()
    netchart = pd.DataFrame(columns = netlist)
    
    # - create a donut plot for visualization
    savefilen=outfilen + "netlist_donutplot.jpg"
    plot_donut(labels=netlist, savefilen=savefilen)
    
    for bc in np.arange(basisCAPs.shape[0]):
        
        mycap = basisCAPs[bc,:]
        cossim_allnet = []
        
        for nt in np.arange(len(netlist)):
            netgt = np.zeros(len(parcnet),)
            netidx = parcnet.index[parcnet['network_id'] == netlist[nt]].tolist()
            netgt[netidx]=1
            # msg = "Generate the map of network " + netlist[nt] #+" : " + str(netgt)
            # logging.info(msg)
            
            cossim=cosine_similarity(a=mycap, b=netgt)
            cossim_allnet.append(cossim)
            # msg = "Cosine similiarty (basis CAP " + str(bc) + ", network " + netlist[nt] + ") = " + str(cossim)
            # logging.info(msg)
        
        # plot polar graph for each basis CAP
        savefilen=outfilen + "_cap" + str(bc) + ".jpg"
        plot_polar_graph(labels = netlist, values=cossim_allnet, bc=bc, savefilen=savefilen)
        
        # - save to output data matrix
        netchart.loc[bc] = cossim_allnet
        del cossim_allnet
        
        msg = "Basis CAP " + str(bc) + " polar graph is generated ..."
        logging.info(msg)
        
    del parcnet
    
    return netchart




# -------------------------------------------
#          Main analysis starts here.
# -------------------------------------------
param.splittype = "Ssplit"
if param.seedtype == "seedbased":
    pthrange = param.sig_threshold_range
elif param.seedtype == "seedfree":
    pthrange = param.randTthresholdrange
basiskrange = param.basis_k_range


parcnet = pd.read_csv(filein.parcelinfo_filen, header=None) 
parcnet.columns=['parcel_id', 'network_name', 'network_id']

for pth in pthrange:
    
    if param.seedtype == "seedbased":
        param.sig_threshold = pth
    elif param.seedtype == "seedfree":
        param.randTthreshold = pth    
    
    for stdk in basiskrange:

        for sp in [1, 2]:
            if sp == 1:
                spdatatag = "split_1"
            elif sp == 2:
                spdatatag = "split_2"

            # -------------- Load BASIS CAPs ----------------- #
            filein.basiscap_dir = filein.outdir + "/P" + str(param.standardTthreshold) + "/" + spdatatag + "/k" + str(stdk) + "/"
            basisCAPs = load_basiscaps(spdatatag=spdatatag, filein=filein, param=param, stdk=stdk)        
            
            # --------- Compare Basis CAPs to pre-defined networks --------- #
            outfilen = filein.outdir + str(stdk) + "_BasisCAP_NetPolarChart_" + spdatatag
            netchart=compare_cap_vs_networks(basisCAPs=basisCAPs, parcnet=parcnet, outfilen=outfilen)
            

plt.close('all')


    
    

# - Notify job completion
msg = "\n========== The jobs are all completed. ==========\n"
logging.info(msg)

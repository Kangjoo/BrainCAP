#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# Created By  : Kangjoo Lee (kangjoo.lee@yale.edu)
# Last Updated: 08/06/2024
# -------------------------------------------------------------------------


# ===============================================================
#     Estimation of Co-Activation Patterns(CAPs) in fMRI data
# ===============================================================


# Imports
from pycap_functions.pycap_frameselection import *
from pycap_functions.pycap_loaddata import *
from kneed import KneeLocator
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import sklearn.cluster
import sklearn.mixture
from scipy import stats
import h5py
import os
import logging
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')


# ======================================================================
#
# -   Parallel process codes to run k-means clustering and generate CAP
#
# ======================================================================

def clusterdata_any(inputdata, filein, param):
    """
    Clusters data according to specified sklearn.cluster function, specified in param.cluster_args
    Must include _method and _variable keys, defining the function and clustering variable.
    Remaining keys must be valid keys for the specified function.
    """
    cluster_args = param.cluster_args.copy()
    cluster_method = cluster_args.pop('_method', None)
    c_var = cluster_args.pop('_variable', None)
    outdir = os.path.join(filein.outpath, cluster_method + "_runs_" + param.spdatatag)
    overwrite = param.overwrite

    if not os.path.exists(outdir):
        try:
            os.makedirs(outdir)
        except:
            pass

    #Won't work here due to multiprocessing
    # elif overwrite == "clear":
    #     os.remove(outdir)
    #     os.makedirs(outdir)

    if isinstance(cluster_args[c_var], list):
        cluster_vals = cluster_args[c_var]
    else:
        cluster_vals = [cluster_args[c_var]]

    for cluster_val in cluster_vals:
        ind_args = cluster_args.copy()
        ind_args[c_var] = cluster_val
        P_outfilen = os.path.join(outdir, f"{param.tag}{c_var}_{cluster_val}_flabel_cluster.csv")
        score_outfilen = os.path.join(outdir, f"{param.tag}{c_var}_{cluster_val}_silhouette_score.csv")

        for file in [P_outfilen, score_outfilen]:
            if os.path.exists(file):
                logging.info(f"PyCap clustering file {file} found")
                if overwrite == 'yes':
                    logging.info("    overwrite 'yes', existing file will be overwritten.")
                else:
                    logging.info("    overwrite 'no', existing file will be saved. Set overwrite 'yes' to re-run.")
                    return None, None
                
        logging.info("============================================")
        logging.info(f"Clustering method: {cluster_method}")
        logging.info(f"    Running with {c_var}={cluster_val}")

        #attempt to load specified clustering method
        try:
            cluster_func = getattr(sklearn.cluster, cluster_method)
        except:
            raise pe.StepError(step="PyCap Clustering",
                                error=f"Incompatible clustering method {cluster_method}! " \
                                "Only functions in sklearn.cluster and sklearn.mixture are compatible.",
                                action=f"Check sklearn documentation for compatible functions.\nCheck sklearn.cluster.{cluster_method} exists.")
        
        #attempt to load specified parameters
        try:
            cluster_obj = cluster_func(**ind_args).fit(inputdata)
        except:
            logging.info(f"FAILED ARGUMENT DICT: {ind_args}")
            raise pe.StepError(step="PyCap Clustering",
                            error=f"Failed adding cluster parameters for {cluster_method}!",
                            action="Check sklearn documentation for valid parameters.")
        
        P = cluster_obj.predict(inputdata)
        score = silhouette_score(inputdata, cluster_obj.labels_)
        score = np.atleast_1d(score)

        # -----------------------------
        #     save output files
        # -----------------------------

        
        df = pd.DataFrame(data=P.astype(float))
        df.to_csv(P_outfilen, sep=' ', header=False, float_format='%d', index=False)
        msg = "Saved cluster labels corresponding to frames in " + P_outfilen
        logging.info(msg)
        logging.info(P.shape)

        
        np.savetxt(score_outfilen, score)
        msg = "Saved silhouette score in " + score_outfilen
        logging.info(msg)
    
    return

def finalcluster2cap_any(inputdata, filein, param):

    
    overwrite = param.overwrite
    pscalar_filen = filein.pscalar_filen
    mask_file = param.mask

    cluster_args = param.cluster_args.copy()
    cluster_method = cluster_args.pop('_method', None)
    c_var = cluster_args.pop('_variable', None)

    indir = os.path.join(filein.outpath, cluster_method + "_runs_" + param.spdatatag)
    outdir = os.path.join(filein.outpath, cluster_method + "_results_" + param.spdatatag)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    msg = "============================================"
    logging.info(msg)
    msg = f"[{cluster_method} clustering based CAP generation]"
    logging.info(msg)

    logging.info(f"Running with cluster variable: {c_var} and the following args:")
    logging.info(cluster_args)

    score_all = []
    for c_val in cluster_args[c_var]:
        score_outfilen = os.path.join(indir, f"{param.tag}{c_var}_{c_val}_silhouette_score.csv")
        score = np.genfromtxt(score_outfilen)
        score_all.append(score)

    if score_all == []:
        logging.info(f"Is this variable empty?: {cluster_args[c_var]}")
        raise pe.StepError(step="post",
                           error="Unable to locate scores due to bad clustering arguments",
                           action="Check your clustering parameter")

    logging.info(score_all)
    kl = KneeLocator(cluster_args[c_var], score_all, curve="convex", direction="decreasing")
    if not kl.elbow:
        logging.info("WARNING: Unable to locate Knee! For best results, please use a wider range of values!")
        logging.info("  KneeLocator failed, setting to largest value")
        final_k = max(cluster_args[c_var])
    else:
        final_k = kl.elbow        
        
    msg = f"The tested {c_var} values are " + str(cluster_args[c_var])
    logging.info(msg)
    msg = "The estimated scores are " + str(score_all)
    logging.info(msg)
    msg = f"The optimal {c_var} is determined as " + str(final_k) + "."
    logging.info(msg)

    msg = f"Load the results from {cluster_method} clustering on the concatenated data matrix ({c_var} = " + str(final_k) + ").."
    logging.info(msg)
    P_outfilen = os.path.join(indir, f"{param.tag}{c_var}_{final_k}_flabel_cluster.csv")
    P = np.genfromtxt(P_outfilen, dtype=int)   
    logging.info(inputdata.shape)
    logging.info(P.shape)
    
    # ------------------------------------------
    # -   Average frames within each cluster
    # ------------------------------------------
    y = 0
    clmean = np.empty([max(P)+1, inputdata.shape[1]])
    for x in range(0, max(P)+1):
        index = np.where(P == x)  # Find the indices of time-frames belonging to a cluster x
        y = y+np.size(index)  # Progress: cumulated number of time-frames assigned to any cluster
        msg = "cluster " + str(x) + ": averaging " + str(np.size(index)) + \
            " frames in this cluster. (progress: " + str(y) + "/" + str(len(P)) + ")"
        logging.info(msg)
        cldata = inputdata[index, :]  # (n_time-points within cluster x space)
        cluster_mean = cldata.mean(axis=1)  # (1 x space)
        cluster_sed = stats.sem(cldata, axis=1)  # (1 x space)
        cluster_z = cluster_mean / cluster_sed
        clmean[x, :] = cluster_z
        if param.savecapimg == "yes":
            # - Save the averaged image within this cluster
            
            if overwrite != "yes" and os.path.exists(outfilen):
                logging.info("Overwrite 'no' and file exists, will not save cap image")
            else:
                if pscalar_filen != None:
                    template = nib.load(pscalar_filen)
                    outfilen = os.path.join(outdir, param.tag + "cluster" + str(x) + "_Zmap.pscalar.nii")
                elif mask_file != None:
                    template = nib.load(mask_file)
                    outfilen = os.path.join(outdir, param.tag +"cluster" + str(x) + "_Zmap.dscalar.nii")
                else:
                    template = None
                if template != None:
                    new_img = nib.Cifti2Image(cluster_z, header=template.header,
                                            nifti_header=template.nifti_header)
                    new_img.to_filename(outfilen)
                    msg = "cluster " + str(x) + ": saved the average map in " + outfilen
                    logging.info(msg)
                else:
                    logging.info("No Parcellation template or Mask file supplied, unable to save CAP image (Requires headers)")

        elif param.savecapimg == "n":
            msg = "cluster " + str(x) + ": do not save the average map."
            logging.info(msg)

    # -----------------------------
    #     save output files
    # -----------------------------

    P_outfilen1 = os.path.join(outdir,f"{param.tag}_{c_var}_" + str(final_k) + "_framelabel_clusterID.hdf5")
    f = h5py.File(P_outfilen1, "w")
    dset1 = f.create_dataset(
        "framecluster", (P.shape[0],), dtype='int', data=P)
    f.close()

    P_outfilen2 = os.path.join(outdir,f"{param.tag}_{c_var}_" + str(final_k) + "_clustermean.hdf5")
    f = h5py.File(P_outfilen2, "w")
    dset1 = f.create_dataset("clmean", (max(P)+1, inputdata.shape[1]), dtype='float32', data=clmean)
    f.close()
    msg = "Saved cluster mean data matrix in " + P_outfilen2
    logging.info(msg)

    return P

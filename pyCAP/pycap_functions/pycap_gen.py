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

def clusterdata(inputdata, filein, param):

    outdir = filein.outpath
    k = param.kmean_k
    kmethod = param.kmean_kmethod
    max_iter = param.kmean_max_iter
    overwrite = param.overwrite

    P_outfilen = outdir + "kmeans_k" + str(k) + "_flabel_cluster.csv"
    score_outfilen = outdir + "kmeans_k" + str(k) + "_" + kmethod + "_score.csv"

    for file in [P_outfilen, score_outfilen]:
        if os.path.exists(file):
            logging.info(f"PyCap file {file} found")
            if overwrite == 'yes':
                logging.info("    overwrite 'yes', existing file will be overwritten.")
            else:
                logging.info("    overwrite 'no', existing file will be saved. Set overwrite 'yes' to re-run.")
                return None, None

    msg = "============================================"
    logging.info(msg)
    msg = "[K-means clustering (k=" + str(k) + ")]"
    logging.info(msg)

    kmeans_kwargs = {"init": "k-means++", "max_iter": max_iter, "random_state": 42}
    kmeans = KMeans(n_clusters=k, n_init='auto', **kmeans_kwargs).fit(inputdata)
    P = kmeans.predict(inputdata)

    if kmethod == "sse":
        score = kmeans.inertia_
    elif kmethod == "silhouette":
        score = silhouette_score(inputdata, kmeans.labels_)
        score = np.atleast_1d(score)

    # -----------------------------
    #     save output files
    # -----------------------------

    
    df = pd.DataFrame(data=P.astype(float))
    df.to_csv(P_outfilen, sep=' ', header=False, float_format='%d', index=False)
    msg = "Saved cluster labels corresponding to frames in " + P_outfilen
    logging.info(msg)

    
    np.savetxt(score_outfilen, score)
    msg = "Saved " + kmethod + " score in " + score_outfilen
    logging.info(msg)

    return P, score

def clusterdata_any(inputdata, filein, param):
    """
    Clusters data according to specified sklearn.cluster function, specified in param.cluster_args
    Must include _method and _variable keys, defining the function and clustering variable.
    Remaining keys must be valid keys for the specified function.aaaaa
    """
    outdir = filein.outpath
    overwrite = param.overwrite

    cluster_args = param.cluster_args.copy()
    cluster_method = cluster_args.pop('_method', None)
    c_var = cluster_args.pop('_variable', None)

    P_outfilen = os.path.join(outdir, f"{cluster_method}_{cluster_args[c_var]}_flabel_cluster.csv")
    score_outfilen = os.path.join(outdir, f"{cluster_method}_{cluster_args[c_var]}_silhouette_score.csv")

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
    logging.info(f"    Running with {c_var}={cluster_args[c_var]}")

    #attempt to load specified clustering method
    try:
        cluster_func = getattr(sklearn.cluster, cluster_method)
    # except:
    #     try:
    #         cluster_func = getattr(sklearn.mixture, cluster_method)
    except:
        raise pe.StepError(step="PyCap Clustering",
                            error=f"Incompatible clustering method {cluster_method}! " \
                            "Only functions in sklearn.cluster and sklearn.mixture are compatible.",
                            action=f"Check sklearn documentation for compatible functions.\nCheck sklearn.cluster.{cluster_method} exists.")
    
    #attempt to load specified parameters
    try:
        cluster_obj = cluster_func(**cluster_args).fit(inputdata)
    except:
        logging.info(f"FAILED ARGUMENT DICT: {cluster_args}")
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

    
    np.savetxt(score_outfilen, score)
    msg = "Saved silhouette score in " + score_outfilen
    logging.info(msg)
    
    return P, score

def finalcluster2cap(inputdata, filein, param):

    outdir = filein.outpath
    pscalar_filen = filein.pscalar_filen
    mink = param.kmean_krange[0]
    maxk = param.kmean_krange[1]+1  # plus 1 to match with python numpy indexing
    kmethod = param.kmean_kmethod
    max_iter = param.kmean_max_iter

    msg = "============================================"
    logging.info(msg)
    msg = "[K-means clustering based CAP generation]"
    logging.info(msg)

    kmeans_kwargs = {"init": "k-means++", "max_iter": max_iter, "random_state": 42}

    # ---------------------------------------
    # -        Determine optimal k
    # ---------------------------------------
    msg = "Determine the number of clusters using " + kmethod + " method.."
    logging.info(msg)
    msg = "Load scores for k in numpy.range (" + str(mink) + ", " + str(maxk-1) + ").."
    logging.info(msg)

    score_all = []
    for k in range(mink, maxk):
        score_outfilen = outdir + "kmeans_k" + str(k) + "_" + kmethod + "_score.csv"
        score = np.genfromtxt(score_outfilen)
        score_all.append(score)
        
    kl = KneeLocator(range(mink, maxk), score_all, curve="convex", direction="decreasing")
    final_k = kl.elbow        
        
    # plt.style.use("fivethirtyeight")
    # plt.plot(range(mink, maxk), score_all)
    # plt.xticks(range(mink, maxk))
    # plt.xlabel("K")
    # plt.ylabel("score" + kmethod)
    # # plt.show(block=False)
    # # plt.pause(1)
    # # plt.close()
    # plt.plot([final_k], score_all[final_k], marker="o", markersize=20, markeredgecolor="red", markerfacecolor="green")
    # savefilen = outdir + "Scores_" + kmethod + "_k" + str(mink) + "to" + str(maxk) + ".png"
    # plt.savefig(savefilen, bbox_inches='tight')
    # msg = "Saved " + savefilen
    # logging.info(msg)    
    msg = "The tested k values are " + str(range(mink, maxk))
    logging.info(msg)
    msg = "The estimated scores are " + str(score_all)
    logging.info(msg)
    msg = "The optimal k is determined as " + str(final_k) + "."
    logging.info(msg)

    
#     # -------------------------------------------
#     # -   Option 1: Perform K-means clustering using optimal k (if P was not saved due to memory limit)
#     # -------------------------------------------
#     msg = "Apply k-means clustering on the concatenated data matrix (k = " + str(final_k) + ").."
#     logging.info(msg)
#     kmeans = KMeans(n_clusters=final_k, **kmeans_kwargs).fit(inputdata)
#     P = kmeans.predict(inputdata)
    
    # -------------------------------------------
    # -   Option 2: Load results from K-means clustering using optimal k
    # -------------------------------------------
    msg = "Load the results from k-means clustering on the concatenated data matrix (k = " + str(final_k) + ").."
    logging.info(msg)
    P_outfilen = outdir + "kmeans_k" + str(k) + "_flabel_cluster.csv"
    P = np.genfromtxt(P_outfilen)   
    
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
        if param.savecapimg == "y":
            # - Save the averaged image within this cluster
            outfilen = outdir + "cluster" + str(x) + "_Zmap.pscalar.nii"
            pscalars = nib.load(pscalar_filen)
            new_img = nib.Cifti2Image(cluster_z, header=pscalars.header,
                                      nifti_header=pscalars.nifti_header)
            new_img.to_filename(outfilen)
            msg = "cluster " + str(x) + ": saved the average map in " + outfilen
            logging.info(msg)
        elif param.savecapimg == "n":
            msg = "cluster " + str(x) + ": do not save the average map."
            logging.info(msg)

    # -----------------------------
    #     save output files
    # -----------------------------

    P_outfilen1 = outdir + "FINAL_k" + str(final_k) + "_framelabel_clusterID.hdf5"
    f = h5py.File(P_outfilen1, "w")
    dset1 = f.create_dataset(
        "framecluster", (P.shape[0],), dtype='int', data=P)
    f.close()

    P_outfilen2 = outdir + "FINAL_k" + str(final_k) + "_clustermean.hdf5"
    f = h5py.File(P_outfilen2, "w")
    dset1 = f.create_dataset("clmean", (max(P)+1, inputdata.shape[1]), dtype='float32', data=clmean)
    f.close()
    msg = "Saved cluster mean data matrix in " + P_outfilen2
    logging.info(msg)

    # -----------------------------
    #  delete intermediate files
    # -----------------------------
    for k in range(mink, maxk):
        score_outfilen = outdir + "kmeans_k" + str(k) + "_" + kmethod + "_score.csv"
        os.remove(score_outfilen)

    return P





def finalcluster2cap_hac(inputdata, filein, param):
    outdir = filein.outpath
    pscalar_filen = filein.pscalar_filen
    final_k = param.kmean_k

    msg = "============================================"
    logging.info(msg)
    msg = "[ Hierarchical agglomerative clustering (Ward's algorithm) for the generation of a basis set of CAPs ] /n"
    logging.info(msg)

    # -------------------------------------------
    # -   Perform Ward's algorithm using k
    # -------------------------------------------
    msg = "Apply hierarchical agglomerative clustering on the concatenated CAP data matrix (#clusters = " + str(
        final_k) + ").."
    logging.info(msg)

    hac = AgglomerativeClustering(
        n_clusters=final_k, affinity='euclidean', linkage='ward')
    P = hac.fit_predict(inputdata)

    # ------------------------------------------
    # -   Average frames within each cluster
    # ------------------------------------------
    y = 0
    clmean = np.empty([max(P)+1, inputdata.shape[1]])
    for x in range(0, max(P)+1):
        index = np.where(P == x)  # Find the indices of time-frames belonging to a cluster x
        y = y+np.size(index)  # Progress: cumulated number of time-frames assigned to any cluster
        msg = "HAC cluster " + str(x) + ": averaging " + str(np.size(index)) + \
            " frames in this cluster. (progress: " + str(y) + "/" + str(len(P)) + ")"
        logging.info(msg)
        cldata = inputdata[index, :]  # (n_time-points within cluster x space)
        cluster_mean = cldata.mean(axis=1)  # (1 x space)
        # Z-score transformation within each CAP
        # cluster_z = stats.zscore(cluster_mean, nan_policy='omit') # (1 x space)
        withincap_mean = cluster_mean.mean(axis=1)
        withincap_std = cluster_mean.std(axis=1,ddof=1)
        cluster_z = (cluster_mean - withincap_mean ) / withincap_std
        msg = "within cap mean = " +  str(withincap_mean) + ", std = " + str(withincap_std)
        logging.info(msg)
        msg = "original cluster mean = " +str(cluster_mean.shape) + str(cluster_mean)
        logging.info(msg)
        msg = "Z-transformed cluster mean = " +str(cluster_z.shape) + str(cluster_z)
        logging.info(msg)
        clmean[x, :] = cluster_z
        if param.savecapimg == "y":
            # - Save the averaged image within this cluster
            outfilen = outdir + "HAC_cluster" + str(x) + "_Zmap.pscalar.nii"
            pscalars = nib.load(pscalar_filen)
            new_img = nib.Cifti2Image(cluster_z, header=pscalars.header,
                                      nifti_header=pscalars.nifti_header)
            new_img.to_filename(outfilen)
            msg = "HAC cluster " + str(x) + ": saved the average map in " + outfilen
            logging.info(msg)
        elif param.savecapimg == "n":
            msg = "HAC cluster " + str(x) + ": do not save the average map."
            logging.info(msg)

    # -----------------------------
    #     save output files
    # -----------------------------

#     P_outfilen = outdir + "FINAL_k" + str(final_k) + "_flabel_clusterP.csv"
#     df = pd.DataFrame(data=P.astype(float))
#     df.to_csv(P_outfilen, sep=' ', header=False, float_format='%d', index=False)
#     msg = "Saved cluster labels corresponding to frames in " + P_outfilen
#     logging.info(msg)

    P_outfilen1 = outdir + "FINAL_k" + str(final_k) + "_framelabel_HACclusterID.hdf5"
    f = h5py.File(P_outfilen1, "w")
    dset1 = f.create_dataset(
        "framecluster", (P.shape[0],), dtype='int', data=P)
    f.close()

    P_outfilen2 = outdir + "FINAL_k" + str(final_k) + "_HACclustermean.hdf5"
    f = h5py.File(P_outfilen2, "w")
    dset1 = f.create_dataset("clmean", (max(P)+1, inputdata.shape[1]), dtype='float32', data=clmean)
    f.close()
    msg = "Saved HAC cluster mean data matrix in " + P_outfilen2
    logging.info(msg)

    return P


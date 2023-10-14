#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Kangjoo Lee (kangjoo.lee@yale.edu)
# Created Date: 01/19/2022
# Last Updated: 04/22/2022
# version ='0.0'
# ---------------------------------------------------------------------------
# ===============================================================
#     Estimation of Co-Activation Patterns(CAPs) in fMRI data
# ===============================================================


# Imports
from pycap_functions.pycap_frameselection import *
from pycap_functions.pycap_loaddata_hcp import *
from kneed import KneeLocator
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
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

    outdir = filein.outdir
    k = param.kmean_k
    kmethod = param.kmean_kmethod
    max_iter = param.kmean_max_iter

    msg = "============================================"
    logging.info(msg)
    msg = "[K-means clustering (k=" + str(k) + ")]"
    logging.info(msg)

    kmeans_kwargs = {"init": "k-means++", "max_iter": max_iter, "random_state": 42}
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs).fit(inputdata)
    P = kmeans.predict(inputdata)

    if kmethod == "sse":
        score = kmeans.inertia_
    elif kmethod == "silhouette":
        score = silhouette_score(inputdata, kmeans.labels_)
        score = np.atleast_1d(score)

    # -----------------------------
    #     save output files
    # -----------------------------

    # P_outfilen = outdir + "kmeans_k" + str(k) + "_flabel_cluster.csv"
    # df = pd.DataFrame(data=P.astype(float))
    # df.to_csv(P_outfilen, sep=' ', header=False, float_format='%d', index=False)
    # msg = "Saved cluster labels corresponding to frames in " + P_outfilen
    # logging.info(msg)

    score_outfilen = outdir + "kmeans_k" + str(k) + "_" + kmethod + "_score.csv"
    np.savetxt(score_outfilen, score)
    msg = "Saved " + kmethod + " score in " + score_outfilen
    logging.info(msg)

    return P, score


def finalcluster2cap(inputdata, filein, param):

    outdir = filein.outdir
    pscalar_filen = filein.pscalar_filen
    mink = param.kmean_krange[0]
    maxk = param.kmean_krange[1]+1  # plus 1 to match with python numpy indexing
    kmethod = param.kmean_kmethod
    max_iter = param.kmean_max_iter

    msg = "============================================"
    logging.info(msg)
    msg = "[K-means clustering based CAP generation]"
    logging.info(msg)

    # kmeans_kwargs = {"init": "random", "max_iter": max_iter}
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
        
#     plt.style.use("fivethirtyeight")
#     plt.plot(range(mink, maxk), score_all)
#     plt.xticks(range(mink, maxk))
#     plt.xlabel("K")
#     plt.ylabel("score" + kmethod)
#     # plt.show(block=False)
#     # plt.pause(1)
#     # plt.close()
#     plt.plot([final_k], score_all[final_k], marker="o", markersize=20, markeredgecolor="red", markerfacecolor="green")
#     savefilen = outdir + "Scores_" + kmethod + "_k" + str(mink) + "to" + str(maxk) + ".png"
#     plt.savefig(savefilen, bbox_inches='tight')
#     msg = "Saved " + savefilen
#     logging.info(msg)    
    msg = "The tested k values are " + str(range(mink, maxk))
    logging.info(msg)
    msg = "The estimated scores are " + str(score_all)
    logging.info(msg)
    msg = "The optimal k is determined as " + str(final_k) + "."
    logging.info(msg)

    # -------------------------------------------
    # -   Perform K-means clustering using k
    # -------------------------------------------
    msg = "Apply k-means clustering on the concatenated data matrix (k = " + str(final_k) + ").."
    logging.info(msg)
    kmeans = KMeans(n_clusters=final_k, **kmeans_kwargs).fit(inputdata)
    P = kmeans.predict(inputdata)

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

    # P_outfilen = outdir + "FINAL_k" + str(final_k) + "_flabel_clusterP.csv"
    # df = pd.DataFrame(data=P.astype(float))
    # df.to_csv(P_outfilen, sep=' ', header=False, float_format='%d', index=False)
    # msg = "Saved cluster labels corresponding to frames in " + P_outfilen
    # logging.info(msg)

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





def finalcluster2cap_pca(inputdata, filein, param):
    outdir = filein.outdir
    pscalar_filen = filein.pscalar_filen
    final_k = param.kmean_k

    msg = "============================================"
    logging.info(msg)
    msg = "[ Principal Component Analysis for generation of a basis set of CAPs ] /n"
    logging.info(msg)

    # -------------------------------------------
    # -   Perform PCA using k
    # -------------------------------------------
    msg = "Apply PCA on the concatenated CAP data matrix (#PC = " + str(final_k) + ").."
    logging.info(msg)

    inputdata = np.transpose(inputdata)
    pca = PCA(n_components=final_k)
    # pca.fit(inputdata)
    inputdata_pca = pca.fit_transform(inputdata)
    msg = "Dimension reduction : " + str(inputdata.shape) + " >> " + str(inputdata_pca.shape)
    logging.info(msg)
    inputdata_pca = np.transpose(inputdata_pca)
    msg = "Transpose data : " + str(inputdata_pca.shape)
    logging.info(msg)
    exp_var_pca = 100 * pca.explained_variance_ratio_
    cum_sum_eigenvalues = np.cumsum(exp_var_pca)
    msg = "Explained variance : " + str(exp_var_pca*100)
    logging.info(msg)

    # Create the visualization plot
    plt.bar(range(0, len(exp_var_pca)), exp_var_pca, alpha=0.5,
            align='center', label='Individual explained variance')
    plt.step(range(0, len(cum_sum_eigenvalues)), cum_sum_eigenvalues,
             where='mid', label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio (%)')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.ylim((0, 100))
    # plt.show()
    savefilen = outdir + "PC_exp_var.png"
    plt.savefig(savefilen, bbox_inches='tight')
    plt.close()

    # ------------------------------------------
    # -   Save PC images
    # ------------------------------------------
    for x in range(final_k):
        mypc = inputdata_pca[[x], ]  # (1 x space)
        msg = "PC " + str(x) + " : " + str(mypc.shape)
        logging.info(msg)

        if param.savecapimg == "y":
            # - Save the averaged image within this cluster
            outfilen = outdir + "PC" + str(x) + "_map.pscalar.nii"
            pscalars = nib.load(pscalar_filen)
            new_img = nib.Cifti2Image(mypc, header=pscalars.header,
                                      nifti_header=pscalars.nifti_header)
            new_img.to_filename(outfilen)
            msg = "PC " + str(x) + ": saved the PC map in " + outfilen
            logging.info(msg)
        elif param.savecapimg == "n":
            msg = "PC " + str(x) + ": do not save the PC map."
            logging.info(msg)

    # -----------------------------
    #     save output files
    # -----------------------------

    P_outfilen2 = outdir + "FINAL_k" + str(final_k) + "_PCA.hdf5"
    f = h5py.File(P_outfilen2, "w")
    dset1 = f.create_dataset(
        "inputdata_pca", (inputdata_pca.shape[0], inputdata_pca.shape[1]), dtype='float32', data=inputdata_pca)
    f.close()
    msg = "Saved PC data matrix in " + P_outfilen2
    logging.info(msg)

    return inputdata_pca


def finalcluster2cap_hac(inputdata, filein, param):
    outdir = filein.outdir
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

    # P_outfilen = outdir + "FINAL_k" + str(final_k) + "_flabel_clusterP.csv"
    # df = pd.DataFrame(data=P.astype(float))
    # df.to_csv(P_outfilen, sep=' ', header=False, float_format='%d', index=False)
    # msg = "Saved cluster labels corresponding to frames in " + P_outfilen
    # logging.info(msg)

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


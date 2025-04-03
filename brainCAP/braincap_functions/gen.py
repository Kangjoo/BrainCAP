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
from braincap_functions.frameselection import *
from braincap_functions.loaddata import *
from braincap_functions.plots import plot_scree
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
                logging.info(f"braincap clustering file {file} found")
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
            raise pe.StepError(step="braincap Clustering",
                                error=f"Incompatible clustering method {cluster_method}! " \
                                "Only functions in sklearn.cluster and sklearn.mixture are compatible.",
                                action=f"Check sklearn documentation for compatible functions.\nCheck sklearn.cluster.{cluster_method} exists.")
        
        #attempt to load specified parameters
        try:
            cluster_obj = cluster_func(**ind_args).fit(inputdata)
        except:
            logging.info(f"FAILED ARGUMENT DICT: {ind_args}")
            raise pe.StepError(step="braincap Clustering",
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

def determine_clusters(filein, param):
    """
    Checks different clusters and determines optimum values
    """

    cluster_args = param.cluster_args.copy()
    cluster_method = cluster_args.pop('_method', None)
    c_var = cluster_args.pop('_variable', None)

    indir = os.path.join(filein.outpath, cluster_method + "_runs_" + param.spdatatag)

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

    return score_all, final_k

def finalcluster2cap_any(inputdata, filein, param, final_k):

    
    overwrite = param.overwrite
    pscalar_filen = filein.pscalar_filen
    mask_file = param.mask

    cluster_args = param.cluster_args.copy()
    cluster_method = cluster_args.pop('_method', None)
    c_var = cluster_args.pop('_variable', None)

    indir = os.path.join(filein.outpath, cluster_method + "_runs_" + param.spdatatag)
    outdir = filein.outpath #os.path.join(filein.outpath, "clustering_results_" + param.spdatatag)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    msg = "============================================"
    logging.info(msg)
    msg = f"[{cluster_method} clustering based CAP generation]"
    logging.info(msg)

    logging.info(f"Running with cluster variable: {c_var} and the following args:")
    logging.info(cluster_args)

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
    P_outfilen2 = os.path.join(outdir,f"{param.tag}clustering_results_{param.spdatatag}.hdf5")
    f = h5py.File(P_outfilen2, "w")
    f.create_dataset("cluster_means", (max(P)+1, inputdata.shape[1]), dtype='float32', data=clmean)
    f.create_dataset(
        "cluster_labels", (P.shape[0],), dtype='int', data=P)
    f.close()
    msg = "Saved cluster mean data matrix in " + P_outfilen2
    logging.info(msg)

    return P, clmean

def create_basis_CAP(inputdata, n_clusters):

    msg = "============================================"
    logging.info(msg)
    msg = "[ Hierarchical agglomerative clustering (Ward's algorithm) for the generation of a basis set of CAPs ] /n"
    logging.info(msg)

    # -------------------------------------------
    # -   Perform Ward's algorithm using k
    # -------------------------------------------
    msg = "Apply hierarchical agglomerative clustering on the concatenated CAP data matrix (#clusters = " + str(
        n_clusters) + ").."
    logging.info(msg)

    #allow other clustering? PCA?
    hac = AgglomerativeClustering(
        n_clusters=n_clusters, affinity='euclidean', linkage='ward')
    P = hac.fit_predict(inputdata)

    # ------------------------------------------
    # -   Average frames within each cluster
    # ------------------------------------------
    y = 0
    clmean = np.empty([max(P)+1, inputdata.shape[1]]) #IS THIS THE RIGHT SHAPE????
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

        #Commented out until solved cap image saving
        # if param.savecapimg == "y":
        #     # - Save the averaged image within this cluster
        #     outfilen = outdir + "HAC_cluster" + str(x) + "_Zmap.pscalar.nii"
        #     pscalars = nib.load(pscalar_filen)
        #     new_img = nib.Cifti2Image(cluster_z, header=pscalars.header,
        #                               nifti_header=pscalars.nifti_header)
        #     new_img.to_filename(outfilen)
        #     msg = "HAC cluster " + str(x) + ": saved the average map in " + outfilen
        #     logging.info(msg)
        # elif param.savecapimg == "n":
        #     msg = "HAC cluster " + str(x) + ": do not save the average map."
        #     logging.info(msg)

    # -----------------------------
    #     save output files
    # -----------------------------

    # P_outfilen = outdir + "FINAL_k" + str(final_k) + "_flabel_clusterP.csv"
    # df = pd.DataFrame(data=P.astype(float))
    # df.to_csv(P_outfilen, sep=' ', header=False, float_format='%d', index=False)
    # msg = "Saved cluster labels corresponding to frames in " + P_outfilen
    # logging.info(msg)

    #file saving should occur in main script

    # P_outfilen1 = outdir + "FINAL_k" + str(final_k) + "_framelabel_HACclusterID.hdf5"
    # f = h5py.File(P_outfilen1, "w")
    # dset1 = f.create_dataset(
    #     "framecluster", (P.shape[0],), dtype='int', data=P)
    # f.close()

    # P_outfilen2 = outdir + "FINAL_k" + str(final_k) + "_HACclustermean.hdf5"
    # f = h5py.File(P_outfilen2, "w")
    # dset1 = f.create_dataset("clmean", (max(P)+1, inputdata.shape[1]), dtype='float32', data=clmean)
    # f.close()
    # msg = "Saved HAC cluster mean data matrix in " + P_outfilen2
    # logging.info(msg)

    return P, clmean

def reorder_R(R, savefilen):
    # R is an n x m correlation matrx (e.g. # of test caps x # of basis caps)
    # the goal is to re-order(re-label) n rows (test caps)
    # to assign a cap index for test cap according to the similarity with basis caps
    # Output: an n x m correlation matrix

    if (R.shape[0] == R.shape[1]):

        msg = "Re-label (re-order) rows of the correlation matrix by sorting test caps (rows) using spatial similarity to basis CAPs."
        logging.info(msg)
        sortcap_match = np.zeros((R.shape[1],))
        for basis_c in np.arange(R.shape[1]):
            sortcap = R[:, basis_c].argsort()
            sortcap = sortcap[::-1]
            sortcap_match[basis_c] = sortcap[0]
        sortcap_match = np.int_(sortcap_match)
        del sortcap

        if np.array_equal(np.unique(sortcap_match), np.unique(np.arange(R.shape[1]))):
            msg = "All test caps are sorted using spatial correlation (r) with basis CAPs."
            logging.info(msg)
        else:
            msg = "There is one or more test caps that are assigned to more than 1 basis CAPs."
            logging.info(msg)

            # ovl_testcapID: Indeces of test caps that are assigned to more than 1 basis CAP
            match, counts = np.unique(sortcap_match, return_counts=True)
            if len(np.where(counts > 1)[0] == 1):
                ovl_testcapID = match[np.where(counts > 1)[0]]

            # Do following: if found one test CAP assigned to more than 1 basis CAP
            # Goal: to compare actual r value of this test CAP with two basis CAPs
            # and assign this test CAP to the basis CAP with higher r
            if len(ovl_testcapID == 1):
                # ovl_basiscapID: Indices of basis CAPs that have assigned to the same test cap
                ovl_basiscapID = np.where(sortcap_match == ovl_testcapID)[0]
                r_tocompare = R[ovl_testcapID, ovl_basiscapID]
                keep_idx = ovl_basiscapID[np.where(r_tocompare == max(r_tocompare))[0]]
                replace_idx = ovl_basiscapID[np.where(r_tocompare == min(r_tocompare))[0]]

                msg = "R(testcap" + str(ovl_testcapID) + ", basiscap" + \
                    str(ovl_basiscapID) + ")=" + str(r_tocompare)
                logging.info(msg)
                msg = "basiscap " + str(keep_idx) + \
                    "should be matched to testcap " + str(ovl_testcapID) + "."
                logging.info(msg)
                msg = "basiscap " + str(replace_idx) + " should be matched to other testcap."
                logging.info(msg)

                missing_idx = np.array(list(set(np.arange(R.shape[1])).difference(match)))
                msg = "Found a test cap without assignment : " + str(missing_idx)
                logging.info(msg)

                sortcap_match[replace_idx] = missing_idx

        if np.array_equal(np.unique(sortcap_match), np.unique(np.arange(R.shape[1]))):
            sorted_R = R[sortcap_match]
            # f, ax = plt.subplots(figsize=(4, 8))
            # plt.subplot(211)
            # ax = sns.heatmap(R, annot=True, linewidths=.5, vmin=-1, vmax=1, cmap=rpal)
            # plt.xlabel('basis CAPs')
            # plt.ylabel('test CAPs')
            # plt.subplot(212)
            # ax = sns.heatmap(sorted_R, annot=True, linewidths=.5, vmin=-1, vmax=1, cmap=rpal)
            # plt.xlabel('basis CAPs')
            # plt.ylabel('Re-ordered test CAPs (new label)')
            # # plt.show()
            # if saveimgflag == 1:
            #     plt.savefig(savefilen, bbox_inches='tight')
            #     msg = "Saved " + savefilen
            #     logging.info(msg)
        else:
            msg = "Cannot save " + savefilen + ": caps were not matched. " + str(sortcap_match)
            logging.info(msg)

    elif (R.shape[0] < R.shape[1]):

        msg = "Re-label (re-order) rows of the correlation matrix by sorting test caps (rows) using spatial similarity to basis CAPs."
        logging.info(msg)
        sortcap_match = np.zeros((R.shape[0],))
        for est_c in np.arange(R.shape[0]):
            sortcap = R[est_c, :].argsort()
            sortcap = sortcap[::-1]
            sortcap_match[est_c] = sortcap[0]
        sortcap_match = np.int_(sortcap_match)
        del sortcap

        if np.array_equal(np.unique(sortcap_match), np.unique(np.arange(R.shape[0]))):
            sorted_R = np.zeros((R.shape))
            for j in np.arange(R.shape[0]):
                idx = sortcap_match[j]
                sorted_R[idx, :] = R[j, :]
            # f, ax = plt.subplots(figsize=(4, 8))
            # plt.subplot(211)
            # ax = sns.heatmap(R, annot=True, linewidths=.5, vmin=-1, vmax=1, cmap=rpal)
            # plt.xlabel('basis CAPs')
            # plt.ylabel('test CAPs')
            # plt.subplot(212)
            # ax = sns.heatmap(sorted_R, annot=True, linewidths=.5, vmin=-1, vmax=1, cmap=rpal)
            # plt.xlabel('basis CAPs')
            # plt.ylabel('Re-ordered test CAPs (new label)')
            # # plt.show()
            # if saveimgflag == 1:
            #     plt.savefig(savefilen, bbox_inches='tight')
            #     msg = "Saved " + savefilen
            #     logging.info(msg)
        else:
            msg = "Cannot save " + savefilen + ": caps were not matched. " + str(sortcap_match)
            logging.info(msg)

    return sorted_R, sortcap_match
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Kangjoo Lee (kangjoo.lee@yale.edu)
# Created Date: 04/04/2022
# Last Updated: 04/22/2022
# version ='0.0'
# ---------------------------------------------------------------------------


# Imports
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import logging
import os
import h5py
import pandas as pd


def cap_occupancy_btwcap(data, param):
    msg = "\n"
    logging.info(msg)
    msg = "    Group-level: Compute between-CAP variance of fractional occupancy .."
    logging.info(msg)

    # Load data
    frame_clusterID = data.frame_clusterID
    n_cap = np.amax(frame_clusterID)
    totTR = len(frame_clusterID)

    # Compute CAP fractional occupancy
    capdur = np.empty((0, 1), int)
    for i in range(n_cap+1):
        idx = np.where(frame_clusterID == i)[0]
        d = np.array([[len(idx)]])
        capdur = np.append(capdur, d, axis=0)
    capdur = capdur.reshape(capdur.shape[0],)
    sum_dur = np.sum(capdur)
    capdur = capdur / totTR * 100
    
    # Get statistics
    bcm_dur = np.mean(capdur)
    bcv_dur = np.std(capdur)
    bccv_dur = np.divide(np.std(capdur), bcm_dur)
    
    # Print results
    msg = "    Group-level each CAP duration: " + str(capdur)
    logging.info(msg)
    msg = "        >> Sum(#group CAP frames) = " + str(sum_dur)
    logging.info(msg)
    msg = "        >> Mean +/- Var (#group CAP frames) = " + \
        str(bcm_dur) + " +/- " + str(bcv_dur)
    logging.info(msg)
    msg = "        >> Coefficient of variation (#group CAP frames) = " + str(bccv_dur)
    logging.info(msg)

    return capdur, bcm_dur, bcv_dur


def cap_occupancy_withincap_btwsubj(data, dataname, filein, param):
    msg = "\n"
    logging.info(msg)
    msg = "    Individual level: Compute within-CAP between-subject variance of fractional occupancy.."
    logging.info(msg)

    # Load data
    frame_clusterID = data.frame_clusterID
    frame_subID = data.frame_subID
    splitlist = data.splitlist
    n_cap = np.amax(frame_clusterID)
    totTR = len(frame_clusterID)

    # Data QC
    msg = "    " + param.splittype + " " + dataname + " data with size: " + str(len(splitlist))
    logging.info(msg)
    # msg = "        >> np.unique(frame_subID) : " + str(np.unique(frame_subID))
    # logging.info(msg)
    # msg = "        >> np.unique(splitlist)   : " + str(np.unique(splitlist))
    # logging.info(msg)

    # Compute CAP fractional occupancy

    wcbss_dur = np.empty((0, 1), dtype=int)
    wcbsm_dur = np.empty((0, 1), dtype=float)
    wcbsv_dur = np.empty((0, 1), dtype=float)

    # Select a CAP each iteration
    
    capdur_ind_allcap = pd.DataFrame(splitlist, columns = ['subID'])
    for i in range(n_cap+1):
    
        # Find a CAP state
        capdur_ind = np.empty((0, 1), int)
        idx = np.where(frame_clusterID == i)[0]
        msg = ""
        logging.info(msg)
        msg = "    - CAP " + str(i) + ": a total of " + str(len(idx)) + " frames."
        logging.info(msg)
        capstate = frame_subID[idx] # array of subID from all frames belonging to this CAP

        # For this CAP, compute occupancy of each subject
        for j in range(len(splitlist)):
            sidx = np.where(capstate == splitlist[j])[0]
            d = np.array([[len(sidx)]])
            capdur_ind = np.append(capdur_ind, d, axis=0)
        capdur_ind = capdur_ind.reshape(capdur_ind.shape[0],)

        # Divide by the total number of frames 
        sum_capdur_ind = np.sum(capdur_ind) 
        # capdur_ind = capdur_ind / sum_capdur_ind * 100
        capdur_ind = capdur_ind / totTR * 100
        
        # get statistics
        mean_capdur_ind = np.mean(capdur_ind)
        var_capdur_ind = np.std(capdur_ind)
        wcbss_dur = np.append(wcbss_dur, sum_capdur_ind)
        wcbsm_dur = np.append(wcbsm_dur, mean_capdur_ind)
        wcbsv_dur = np.append(wcbsv_dur, var_capdur_ind)
        
        # Save individual data for this CAP
        capdur_ind_tmp = pd.DataFrame(capdur_ind, columns = [str(i)])
        capdur_ind_allcap = pd.concat([capdur_ind_allcap, capdur_ind_tmp], axis=1)        
        
        # Print results
        msg = "        >> #individual CAP frames = " + str(len(capdur_ind_allcap.columns))
        logging.info(msg)
        msg = "        >> Sum(#individual CAP frames) = " + str(sum_capdur_ind)
        logging.info(msg)
        msg = "        >> Mean +/ Var(#individual CAP frames) = " + \
            str(mean_capdur_ind) + " +/- "+str(var_capdur_ind)
        logging.info(msg)
    

    msg= "\nTable: Individual Occupancy (#frames) : \n" + str(capdur_ind_allcap)
    logging.info(msg)
    

    return capdur_ind_allcap, wcbsm_dur, wcbsv_dur




def cap_dwelltime_withincap_withinsubj(data, dataname, filein, param):
    msg = "\n"
    logging.info(msg)
    msg = "    Individual level: Compute within-CAP within-subject variance of dwell time.."
    logging.info(msg)

    # Load data
    frame_clusterID = data.frame_clusterID
    frame_subID = data.frame_subID
    splitlist = data.splitlist
    n_cap = np.amax(frame_clusterID)

    # Data QC
    msg = "    " + param.splittype + " " + dataname + " data with size: " + str(len(splitlist))
    logging.info(msg)
    # msg = "        >> np.unique(frame_subID) : " + str(np.unique(frame_subID))
    # logging.info(msg)
    # msg = "        >> np.unique(splitlist)   : " + str(np.unique(splitlist))
    # logging.info(msg)
    msg= "The total number of CAP states is " + str(len(frame_clusterID)) + " : " + str(frame_clusterID)
    logging.info(msg)
    msg= "The total number of Subject ID is " + str(len(frame_subID)) + " : " + str(frame_subID)
    logging.info(msg) 
    msg = "\n\n"
    logging.info(msg)

    # Initialize output variables (an array of #subjects x #caps)
    capdwt_wsm_allcap = pd.DataFrame()
    capdwt_wsv_allcap = pd.DataFrame()
    colwarg = np.arange(n_cap+1).tolist()
    colwarg = [str(x) for x in colwarg] 

    
    # Compute CAP dwell times in each subject
    for j in range(len(splitlist)):
        sidx = np.where(frame_subID == splitlist[j])[0]
        scap = frame_clusterID[sidx]
        # msg= "Subject ID: " + str(splitlist[j]) + " has " + str(len(scap)) + " CAP states : " + str(scap)
        # logging.info(msg)
        
        # initialize for each subject
        wcwsm_dwt = np.empty((0, 1), dtype=float)
        wcwsv_dwt = np.empty((0, 1), dtype=float)
        for i in range(n_cap+1):
            # wcwsm_dwt: within-cap within-subject mean of dwell times
            # wcwsv_dwt: within-cap within-subject variance of dwell times
            wcwsm, wcwsv = compute_dwell_time(capIDarray=scap, capID=i)
        
            # collect over CAPs
            wcwsm_dwt = np.append(wcwsm_dwt, wcwsm)
            wcwsv_dwt = np.append(wcwsv_dwt, wcwsv)
        
        # collect results from all subjects
        df_info = pd.DataFrame(data=wcwsm_dwt.reshape(-1, len(wcwsm_dwt)), columns=colwarg)
        df_info.insert(0,"subID", splitlist[j])
        capdwt_wsm_allcap = pd.concat([capdwt_wsm_allcap, df_info], axis=0,ignore_index=True)
        
        df_info2 = pd.DataFrame(data=wcwsv_dwt.reshape(-1, len(wcwsv_dwt)), columns=colwarg)
        df_info2.insert(0,"subID", splitlist[j])
        capdwt_wsv_allcap = pd.concat([capdwt_wsv_allcap, df_info2], axis=0,ignore_index=True)

    return capdwt_wsm_allcap, capdwt_wsv_allcap



def compute_dwell_time(capIDarray, capID):

    # Binarize the capIDarray to select this cap
    capIDarray = (capIDarray == capID).astype(np.int_)
    cap_timestamps = [i for i, x in enumerate(capIDarray) if x == 1]
    
    dwt_array = np.empty((0, 1), dtype=float)
    counter = 1
    for c in np.arange(1,len(cap_timestamps)):
        if (cap_timestamps[c] == cap_timestamps[c-1] + 1):
            counter +=1
            if (c == len(cap_timestamps)-1):
                dwt_array = np.append(dwt_array, counter)
        elif (cap_timestamps[c] != cap_timestamps[c-1] + 1):   
            dwt_array = np.append(dwt_array, counter)
            counter = 1
            if (c == len(cap_timestamps)-1):
                dwt_array = np.append(dwt_array, counter)            
    
    wcwsm = np.mean(dwt_array)
    wcwsv = np.std(dwt_array)
    # msg="    >> Estimated dwell time for CAP " + str(capID) + ": " + str(wcwsm) + "+/-" + str(wcwsv) +" (mean +/- SD)"
    # logging.info(msg)
    
    return wcwsm, wcwsv
    
    






#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# Created By  : Kangjoo Lee (kangjoo.lee@yale.edu)
# Last Updated: 08/06/2024
# -------------------------------------------------------------------------


# ===============================================================
#        Spatiotemporal Frame Selection for CAPs analysis
# ===============================================================

# Imports
import os
import numpy as np
import pandas as pd
import logging
import h5py
from pycap_functions.pycap_loaddata import load_groupdata_motion, load_groupdata
import pycap_functions.pycap_exceptions as pe
import pycap_functions.pycap_utils as utils

from memory_profiler import profile

# @profile
def prep_scrubbed(inputdata, labeldata, seeddata, filein, param):
    # inputdata: (concantenated time points x space) matrix of whole brain time-course

    outdir = filein.datadir
    sublist = filein.sublist

    seed_args = param.seed_args
    if not seed_args:
        seed_based = None
    else:
        seed_based = seed_args['seed_based'].lower() == "yes"

    msg = "============================================"
    logging.info(msg)
    msg = "[Temporal frame selection]"
    logging.info(msg)
    logging.info(f"Running with seed_based '{seed_based}'")

    labeldata_fsel_outfilen = os.path.join(filein.datadir, param.tag + param.spdatatag + ".hdf5")

    if os.path.exists(labeldata_fsel_outfilen):
        inputdata_fsel, labeldata_fsel = load_groupdata(filein, param)

    else:

        # ------------------------------------------------------------------------
        #   Select datapoints if split data
        # ------------------------------------------------------------------------

        if param.randTthreshold != 100:
            flag_sp = frameselection_Tsubsample(inputdata.shape[0], filein, param)

        # ------------------------------------------------------------------------
        #   Motion scrubbing
        # ------------------------------------------------------------------------

        flag_scrubbed_all, motion_metric = frameselection_motion(filein, param, sublist, inputdata)

        #Flag based on seed, if no seed will flag all as usable
        if seed_based:
            flag_events_all = frameselection_seedactivation(seeddata, seed_args, param, sublist, labeldata)
        else:
            flag_events_all = np.ones_like(flag_scrubbed_all)


        # ------------------------------------------------------------------------
        #   Combine all frame flags (1: select, 0: remove)
        # ------------------------------------------------------------------------
        if 'flag_sp' in locals():
            flag_comb = np.array((flag_scrubbed_all + flag_events_all + flag_sp) > 2, dtype=int)
        else:
            flag_comb = np.array((flag_scrubbed_all + flag_events_all) > 1, dtype=int)
        flag_all = np.array(np.where(flag_comb == 1))
        flag_all_idx = flag_all.tolist()
        # - QC
        framenum_comb = np.size(flag_all_idx)
        percent_comb = (framenum_comb) / np.shape(flag_comb)[0] * 100
        msg = "Combined frame selection: " + str(framenum_comb) + "/" + str(np.shape(flag_comb)[0]) + " frame(s) (" + str(
            round(percent_comb, 2)) + "% of total time-frames) has(ve) been selected finally."
        logging.info(msg)

        # ------------------------------------------------------------------------
        #   Final frame selection
        # ------------------------------------------------------------------------

        inputdata_fsel = inputdata[tuple(flag_all_idx)]
        labeldata_fsel = labeldata[tuple(flag_all_idx)]
        msg = ">> Output: a (" + str(inputdata_fsel.shape[0]) + " x " + str(
            inputdata_fsel.shape[1]) + ") array of (selected time-frames x space)."
        logging.info(msg)

        # ------------------------------------------------------------------------
        #   Save output files
        # ------------------------------------------------------------------------

        f = h5py.File(labeldata_fsel_outfilen, "w")
        dset1 = f.create_dataset(
            "sublabel_all", (labeldata_fsel.shape[0],), dtype='int', data=utils.id2index(labeldata_fsel,filein.sublistfull))
        dset2 = f.create_dataset(
            "data_all", (inputdata_fsel.shape[0],inputdata_fsel.shape[1]), dtype='float32', data=inputdata_fsel)
        f.close()
        msg = "Saved subject labels corresponding to selected frames in " + labeldata_fsel_outfilen
        logging.info(msg)

    return inputdata_fsel, labeldata_fsel

# @profile
def motion_qc(filein, param):
    # inputdata: (concantenated time points x space) matrix of whole brain time-course

    outdir = filein.datadir
    sublist = filein.sublist

    msg = "============================================"
    logging.info(msg)
    msg = "[Motion Quality Control]"
    logging.info(msg)

    # ------------------------------------------------------------------------
    #   Motion data and scrubbing flag
    # ------------------------------------------------------------------------

    flag_scrubbed_all, motion_metric = frameselection_motion(filein, param, sublist)


    return flag_scrubbed_all, motion_metric

    
#############################################################################
#                               Sub-functions                               #
#############################################################################


def frameselection_Tsubsample(allTdim, filein, param):
    import random
    arr = np.arange(0, allTdim, dtype=int)
    tpsize = round(allTdim*param.randTthreshold/100)
    splist = np.random.choice(arr, size=tpsize, replace=False)
    splist.sort()
    param.splist = splist.tolist()
    msg = "Select " + str(param.randTthreshold) + \
        " % random time-frames = " + str(len(param.splist))
    logging.info(msg)
    msg = str(param.splist)
    logging.info(msg)

    flag_sp = np.zeros(allTdim)
    flag_sp[param.splist] = 1
    # - QC
    framenum_splitted = np.size(np.where(flag_sp == 1))
    percent_splitted = (framenum_splitted) / np.shape(flag_sp)[0] * 100
    msg = "split data: " + str(framenum_splitted) + "/" + str(np.shape(flag_sp)[0]) + \
        " frame(s) (" + str(round(percent_splitted, 2)) + \
        "% of total time-frames) selected."
    logging.info(msg)

    # - Save
    f = h5py.File(filein.Tsubsample_filen, "w")
    dset1 = f.create_dataset(
        "flag_sp", (flag_sp.shape[0],), dtype='int', data=flag_sp)
    f.close()

    msg = "Saved the temporal subsampling data: " + filein.Tsubsample_filen
    logging.info(msg)

    return flag_sp


# @profile
def frameselection_motion(filein, param, sublist, inputdata):
    scrubbing = param.scrubbing
    if scrubbing == "yes":
        motion_type = param.motion_type
        motion_threshold = param.motion_threshold

    flag_scrubbed_all = np.array([])
    if scrubbing == "yes":

        msg = "motion scrubbing: find time-frames with excessive motion (" + \
            motion_type + ">" + str(motion_threshold) + "mm).."
        logging.info(msg)
        # -load motion data (time points x n_subjects)
        motion_metric = load_groupdata_motion(filein=filein, param=param)
        # - Select frames with low motion (1: select, 0: scrubbed)
        flag_scrubbed = np.array(motion_metric < motion_threshold, dtype=int)

    elif scrubbing == "no":

        msg = "motion scrubbing: not performed."
        logging.info(msg)
        # - Select all frames
        indTlen = int(np.shape(inputdata)[0] / len(sublist))
        flag_scrubbed = np.ones((indTlen, len(sublist)), dtype=bool)

    # - Concatenate individual flags
    for n_sub in range(0, len(sublist)):
        flag_scrubbed_ind = flag_scrubbed[:, n_sub]
        flag_scrubbed_all = np.concatenate((flag_scrubbed_all, flag_scrubbed_ind), axis=0)

    # - QC
    framenum_scrubbed = np.size(np.where(flag_scrubbed_all == 1))
    percent_scrubbed = (framenum_scrubbed) / np.shape(flag_scrubbed_all)[0] * 100
    msg = "motion scrubbing: " + str(framenum_scrubbed) + "/" + \
        str(np.shape(flag_scrubbed_all)[0]) + " frame(s) (" + str(
        round(percent_scrubbed, 2)) + "% of total time-frames) has(ve) passed motion screening."
    logging.info(msg)

    return flag_scrubbed_all, motion_metric

def frameselection_seedactivation(seeddata, seed_args, param, sublist, labeldata):
    eventtype = seed_args['event_type']
    sig_thresholdtype = seed_args['threshold_type']  # "T" "P"
    sig_threshold = seed_args['threshold']

    # Average
    # --------------------------------------------------------------------
    if sig_thresholdtype == "T":
        msg = eventtype + " detection: find time-frames with seed " + str(seed_args['seed']) + \
            " signals above absolute threshold " + \
            sig_thresholdtype + "=" + str(sig_threshold) + ".."
        logging.info(msg)
        # - Amplitude based thresholding for each individual
        # - Select frames with (de)activation (1: select, 0: remove)
        if eventtype == "activation":
            flag_events_all = np.array(seeddata > sig_threshold, dtype=int)
        elif eventtype == "deactivation":
            flag_events_all = np.array(seeddata < -sig_threshold, dtype=int)
    # --------------------------------------------------------------------
    elif sig_thresholdtype == "P":
        flag_events_all = np.zeros_like(seeddata)
        msg = eventtype + " detection: find time-frames with signals above percentage threshold " + \
        sig_thresholdtype + "=" + str(sig_threshold) + "%.."
        logging.info(msg)
        #Thresholding based on the individual
        for sub in sublist:
            #get data indices belonging to the subject
            indices = np.where(labeldata==sub)
            sub_data = seeddata[indices]
            P_tp = sub_data.shape[0] * sig_threshold/100  # number of time-samples to be selected

            # - Percentage based thresholding for each individual
            # - Select frames with (de)activation (1: select, 0: remove)
            if eventtype == "activation":
                Psort = np.flip(np.sort(sub_data))  # sort in descending order
                flag_events_all[indices] = np.array(sub_data >= Psort[(int(P_tp))], dtype=int)
            elif eventtype == "deactivation":
                Psort = np.sort(sub_data)  # sort in ascending order
                flag_events_all[indices] = np.array(sub_data <= Psort[(int(P_tp))], dtype=int)
    flag_events_all = flag_events_all.flatten()
    # --------------------------------------------------------------------

    # - QC
    framenum_events = np.size(np.where(flag_events_all == 1))
    percent_events = (framenum_events) / np.shape(flag_events_all)[0] * 100
    msg = eventtype + " detection: " + str(framenum_events) + "/" + str(np.shape(flag_events_all)[0]) + " frame(s) (" + str(
        round(percent_events, 2)) + "% of total time-frames) has(ve) signals above " + sig_thresholdtype + " threshold."
    logging.info(msg)

    #raise

    return flag_events_all



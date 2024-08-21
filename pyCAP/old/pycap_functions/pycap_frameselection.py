#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Kangjoo Lee (kangjoo.lee@yale.edu)
# Created Date: 01/19/2022
# Last Updated: 05/17/2022
# version ='0.0'
# ---------------------------------------------------------------------------
# ===============================================================
#        Spatiotemporal Frame Selection for CAPs analysis
# ===============================================================

# Imports
import os
import numpy as np
import pandas as pd
import logging
import h5py
from pycap_functions.pycap_loaddata_hcp import load_hpc_groupdata_motion
from memory_profiler import profile




# @profile
def frameselection_wb(inputdata, labeldata, filein, param):
    # inputdata: (concantenated time points x space) matrix of whole brain time-course

    outdir = filein.outdir
    sublist = filein.sublist

    msg = "============================================"
    logging.info(msg)
    msg = "[Temporal frame selection]"
    logging.info(msg)

    # ------------------------------------------------------------------------
    #   Select datapoints if split data
    # ------------------------------------------------------------------------

    try:
        flag_sp = frameselection_Tsubsample(inputdata.shape[0], filein, param)
    except:
        msg = "No timepoints splitted."
        logging.info(msg)

    # ------------------------------------------------------------------------
    #   Motion scrubbing
    # ------------------------------------------------------------------------

    flag_scrubbed_all = frameselection_motion(filein, param, sublist)

    # ------------------------------------------------------------------------
    #   Combine all frame flags (1: select, 0: remove)
    # ------------------------------------------------------------------------

    if 'flag_sp' in locals():
        flag_comb = np.array((flag_scrubbed_all + flag_sp) > 1, dtype=int)
    else:
        flag_comb = np.array((flag_scrubbed_all) > 0, dtype=int)
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

    # labeldata_fsel_outfilen = outdir + "flabel_subID.csv"
    # df = pd.DataFrame(data=labeldata_fsel.astype(float))
    # df.to_csv(labeldata_fsel_outfilen, sep=' ', header=False, float_format='%d', index=False)
    labeldata_fsel_outfilen = outdir + "Framelabel_subID.hdf5"

    # if os.path.exists(labeldata_fsel_outfilen):
    if (param.kmean_k == param.kmean_krange[0]):
        f = h5py.File(labeldata_fsel_outfilen, "w")
        dset1 = f.create_dataset(
            "labeldata_fsel", (labeldata_fsel.shape[0],), dtype='int', data=labeldata_fsel)
        f.close()
        msg = "Saved subject labels corresponding to selected frames in " + labeldata_fsel_outfilen
        logging.info(msg)
    else:
        msg = "Do not save Frame-wise subject labels. The Framelabel file may exist: " + labeldata_fsel_outfilen
        logging.info(msg)

    return inputdata_fsel, labeldata_fsel




# @profile
def frameselection_wb_daylabel(inputdata, daydata, filein, param):
    # inputdata: (concantenated time points x space) matrix of whole brain time-course

    outdir = filein.outdir
    sublist = filein.sublist

    msg = "============================================"
    logging.info(msg)
    msg = "[Temporal frame selection]"
    logging.info(msg)

    # ------------------------------------------------------------------------
    #   Select datapoints if split data
    # ------------------------------------------------------------------------

    try:
        flag_sp = frameselection_Tsubsample(inputdata.shape[0], filein, param)
    except:
        msg = "No timepoints splitted."
        logging.info(msg)

    # ------------------------------------------------------------------------
    #   Motion scrubbing
    # ------------------------------------------------------------------------

    flag_scrubbed_all = frameselection_motion(filein, param, sublist)

    # ------------------------------------------------------------------------
    #   Combine all frame flags (1: select, 0: remove)
    # ------------------------------------------------------------------------

    if 'flag_sp' in locals():
        flag_comb = np.array((flag_scrubbed_all + flag_sp) > 1, dtype=int)
    else:
        flag_comb = np.array((flag_scrubbed_all) > 0, dtype=int)
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

    daydata_fsel = daydata[tuple(flag_all_idx)]
    
    # ------------------------------------------------------------------------
    #   Save output files
    # ------------------------------------------------------------------------

    # labeldata_fsel_outfilen = outdir + "flabel_subID.csv"
    # df = pd.DataFrame(data=labeldata_fsel.astype(float))
    # df.to_csv(labeldata_fsel_outfilen, sep=' ', header=False, float_format='%d', index=False)
    daydata_fsel_outfilen = outdir + "Framelabel_day.hdf5"


    f = h5py.File(daydata_fsel_outfilen, "w")
    dset1 = f.create_dataset(
        "daydata_fsel", (daydata_fsel.shape[0],), dtype='int', data=daydata_fsel)
    f.close()
    msg = "Saved day labels corresponding to selected frames in " + daydata_fsel_outfilen
    logging.info(msg)


    return daydata_fsel



#############################################################################
#                               Sub-functions                               #
#############################################################################


def frameselection_Tsubsample(allTdim, filein, param):
    filein.Tsubsample_filen = filein.outdir + "Tsubsample_" + param.seedIDname + "_P" + \
        str(param.randTthreshold) + "_" + param.unit + \
        "_" + param.gsr + "_" + param.spdatatag + ".hdf5"

    if os.path.exists(filein.Tsubsample_filen):
        msg = "File exists. Load the list of random temporal subsampling data file: " + filein.Tsubsample_filen
        logging.info(msg)

        f = h5py.File(filein.Tsubsample_filen, 'r')
        flag_sp = f['flag_sp'][:]

    else:
        msg = "File does not exist. Generate random temporal subsampling."
        logging.info(msg)

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
def frameselection_motion(filein, param, sublist):
    scrubbing = param.scrubbing
    if scrubbing == "y":
        motion_type = param.motion_type
        motion_threshold = param.motion_threshold

    flag_scrubbed_all = np.array([])
    if scrubbing == "y":

        msg = "motion scrubbing: find time-frames with excessive motion (" + \
            motion_type + ">" + str(motion_threshold) + "mm).."
        logging.info(msg)
        # -load motion data (time points x n_subjects)
        motion_metric = load_hpc_groupdata_motion(filein=filein, param=param)
        # - Select frames with low motion (1: select, 0: scrubbed)
        flag_scrubbed = np.array(motion_metric < motion_threshold, dtype=int)

    elif scrubbing == "n":

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

    return flag_scrubbed_all



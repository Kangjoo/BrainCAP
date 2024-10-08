#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Kangjoo Lee (kangjoo.lee@yale.edu)
# Created Date: 01/19/2022
# Last Updated: 04/22/2022
# version ='0.0'
# ---------------------------------------------------------------------------
# ===============================================================
#                           Load fMRI data
# ===============================================================


# Imports
import numpy as np
import nibabel as nib
import logging
import os
from scipy import stats
import h5py
# from memory_profiler import profile
# @profile


def load_norm_subject_wb(dataname):
    # an individual (time points x space) matrix
    data = nib.load(dataname).get_fdata(dtype=np.float32)
    zdata = stats.zscore(data, axis=0)  # Normalize each time-series
    del data
    return zdata



def load_groupdata_wb(filein, param):
    homedir = filein.sessions_folder
    sublist = filein.sublist
    fname = filein.fname
    gsr = param.gsr
    unit = param.unit
    #sdim = param.sdim
    #tdim = param.tdim

    msg = "============================================"
    logging.info(msg)
    msg = "[whole-brain] Load " + unit + \
        "-level time-series data preprocessed with " + gsr + ".."
    logging.info(msg)
    
    #set up dimensions for subject concatenated array
    tdim = 0
    sdim = 0
    for idx, subID in enumerate(sublist):
        # - Load fMRI data
        dataname = os.path.join(homedir, str(subID), fname)
        #If data was concatenated using pycap_concatenate, dimensions are saved
        if os.path.exists(dataname + ".npy"):
            dshape = np.load(dataname + ".npy")
        #Otherwise, must load file and get dim directly. Processing inefficent but should be more memory efficient
        else:
            dshape = nib.load(dataname).get_fdata(dtype=np.float32).shape
        tdim += dshape[0]
        if sdim == 0:
            sdim = dshape[1]
        else:
            if sdim != dshape[1]:
                exit() #ERROR

    data_all = np.empty((tdim, sdim), dtype=np.float32)
    sublabel_all = np.empty((tdim, ), dtype=np.int32)
    ptr = 0
    for idx, subID in enumerate(sublist):
        # - Load fMRI data
        dataname = os.path.join(homedir, str(subID), fname)
        zdata = load_norm_subject_wb(dataname)
        data_all[ptr:ptr+zdata.shape[0], :] = zdata
        # - Create subject label
        subid_v = [idx] * zdata.shape[0]
        subid_v = np.array(subid_v)
        sublabel_all[ptr:ptr+zdata.shape[0], ] = subid_v
        # - Update/delete variables
        ptr += zdata.shape[0]

        msg = "(Subject " + str(idx) + ")" + dataname + " " + \
            ", data:" + str(zdata.shape) + ", label:" + str(subid_v.shape)
        logging.info(msg)

        del zdata, subid_v

    msg = ">> Output: a (" + str(data_all.shape[0]) + " x " + \
        str(data_all.shape[1]) + ") array of (group concatenated time-series x space)."
    logging.info(msg)
    return data_all, sublabel_all


def load_groupdata_wb_usesaved(filein, param):
    # filein.groupdata_wb_filen = filein.datadir + "hpc_groupdata_wb_" + \
    #     param.unit + "_" + param.gsr + "_" + param.spdatatag + ".hdf5"
    filein.groupdata_wb_filen = os.path.join(filein.datadir,  "hpc_groupdata_wb_" + \
        param.unit + "_" + param.gsr + "_" + param.spdatatag + ".hdf5")
    if os.path.exists(filein.groupdata_wb_filen):
        msg = "File exists. Load concatenated fMRI/label data file: " + filein.groupdata_wb_filen
        logging.info(msg)

        f = h5py.File(filein.groupdata_wb_filen, 'r')
        data_all = f['data_all']
        sublabel_all = np.asarray([filein.sublist[idx] for idx in f['sublabel_all']])

    else:
        msg = "File does not exist. Load individual whole-brain fMRI data."
        logging.info(msg)

        data_all, sublabel_all = load_groupdata_wb(filein=filein, param=param)
        f = h5py.File(filein.groupdata_wb_filen, "w")
        dset1 = f.create_dataset(
            "data_all", (data_all.shape[0], data_all.shape[1]), dtype='float32', data=data_all)
        dset2 = f.create_dataset(
            "sublabel_all", (sublabel_all.shape[0],), dtype='int', data=sublabel_all)
        f.close()

        msg = "Saved the concatenated fMRI/label data: " + filein.groupdata_wb_filen
        logging.info(msg)
    return data_all, sublabel_all.astype(int)



def load_groupdata_motion(filein, param):
    # load motion parameters estimated using QuNex
    # https://bitbucket.org/oriadev/qunex/wiki/UsageDocs/MovementScrubbing
    # Use outputs from the command `general_compute_bold_list_stats`
    # In (filename).bstats,the columns may be provided in the following order:
    # frame number, n, m, min, max, var, sd, dvars, dvarsm, dvarsme, fd.

    homedir = filein.sessions_folder
    sublist = filein.sublist
    motion_type = param.motion_type

    motion_data_all = np.array([])
    subiter = 1
    for subID in sublist:

        # ------------------------------------------
        #       Individual motion data analysis
        # ------------------------------------------
        msg = "     (Subject " + str(subiter) + \
            ") load frame-wise motion estimates(" + motion_type + ").."
        logging.info(msg)
        motion_data_ind = np.array([])

        #runiter = 1
        #for n_run in run_order:

            # - Load motion estimates in each run from QuNex output
        # motion_data_filen = homedir + str(subID) + \
        #     "/images/functional/movement/bold" + str(n_run) + ".bstats"
        motion_data_filen = os.path.join(homedir, str(subID), filein.motion_file)
        motion_dlist = np.genfromtxt(motion_data_filen, names=True)
        idx = np.where(np.char.find(motion_dlist.dtype.names, motion_type) == 0)
        motion_data_ind = np.genfromtxt(motion_data_filen, skip_header=1, usecols=idx[0])

        # - Remove dummy time-frames
        #motion_data_run = np.delete(motion_data_run, range(n_dummy), 0)

        # - Concatenate individual runs ( ((n_runs) x n_timeframes) x 1 )
        # if runiter == 1:
        #     motion_data_ind = motion_data_run
        # elif runiter > 1:
        #     motion_data_ind = np.concatenate((motion_data_ind, motion_data_run), axis=0)

        #runiter = runiter+1

        # ------------------------------------------
        #       Stack individual motion data
        # ------------------------------------------

        if subiter == 1:
            motion_data_all = motion_data_ind.reshape(-1, 1)
        elif subiter > 1:
            motion_data_all = np.concatenate(
                (motion_data_all, motion_data_ind.reshape(-1, 1)), axis=1)
        subiter = subiter+1

    msg = "     >> Output: a (" + str(motion_data_all.shape[0]) + " x " + str(
        motion_data_all.shape[1]) + ") array of (concatenated " + \
        motion_type + " timeseries x n_subjects)."
    logging.info(msg)

    return motion_data_all



def load_groupdata_wb_daylabel(filein, param):
    homedir = filein.sessions_folder
    sublist = filein.sublist
    fname = filein.fname
    gsr = param.gsr
    unit = param.unit
    sdim = param.sdim
    tdim = param.tdim

    msg = "============================================"
    logging.info(msg)
    msg = "[whole-brain] Load " + unit + \
        "-level time-series data preprocessed with " + gsr + ".."
    logging.info(msg)

    data_all = np.empty((len(sublist) * tdim, sdim), dtype=np.float32)
    sublabel_all = np.empty((len(sublist) * tdim, ), dtype=np.int32)
    daylabel_all = np.empty((len(sublist) * tdim, ), dtype=np.int32)
    ptr = 0
    for idx, subID in enumerate(sublist):
        # - Load fMRI data
        dataname = os.path.join(homedir, str(subID), "images", "functional", fname)
        zdata = load_norm_subject_wb(dataname)
        data_all[ptr:ptr+zdata.shape[0], :] = zdata
        # - Create subject label
        subid_v = [subID] * zdata.shape[0]
        subid_v = np.array(subid_v)
        sublabel_all[ptr:ptr+zdata.shape[0], ] = subid_v
        # - Creat day label
        day_v = np.empty(zdata.shape[0]); day_v.fill(1)
        runlen=int(zdata.shape[0]/2)
        day_v[runlen:] = 2 
        daylabel_all[ptr:ptr+zdata.shape[0], ] = day_v
        # - Update/delete variables
        ptr += zdata.shape[0]

        msg = "(Subject " + str(idx) + ")" + dataname + " " + \
            ", data:" + str(zdata.shape) + ", subject label:" + str(subid_v.shape) + \
            ", day label:" + str(day_v.shape)
        logging.info(msg)

        del zdata, subid_v

    msg = ">> Output 1: a (" + str(data_all.shape[0]) + " x " + \
        str(data_all.shape[1]) + ") array of (group concatenated time-series x space)."
    logging.info(msg)
    msg = ">> Output 2: a " + str(sublabel_all.shape) + " array of (group concatenated subject label)."
    logging.info(msg)
    msg = ">> Output 3: a " + str(daylabel_all.shape[0]) + " array of (group concatenated day label)."
    logging.info(msg)    
    return data_all, sublabel_all, daylabel_all

def concatenate_data(files, ndummy):

    image_header = nib.load(files[0]).header
    im_axis = image_header.get_axis(0)

    images_data = np.vstack([nib.load(file).get_fdata()[ndummy:] for file in files])

    ax_0 = nib.cifti2.SeriesAxis(start = im_axis.start, step = im_axis.step, size = images_data.shape[0]) 
    ax_1 = image_header.get_axis(1)
    new_h = nib.cifti2.Cifti2Header.from_axes((ax_0, ax_1))
    conc_image = nib.Cifti2Image(images_data, new_h)
    conc_image.update_headers()
    return conc_image

def concatenate_motion(motionpaths, ndummy):
    motion_conc = []
    first=True
    for motionpath in motionpaths:
        with open(motionpath, 'r') as f:
            motion = f.read().splitlines()

        if first:
            motion_conc.append(motion[0])
            first=False

        del motion[0]
        #Remove commented and dummy frames
        motion_conc += [line for line in motion if line[0] != "#"][ndummy:]

    return motion_conc

def parse_slist(sessionsfile):
    sessions=[]
    with open(sessionsfile, 'r') as f:
        sessionslist = f.read().splitlines()
    for line in sessionslist:
        elements = line.strip().split(':')
        if len(elements) == 1:
            sessions.append(elements[0].strip())
        elif len(elements) == 2:
            if elements[0] in ['subject id', 'session id']:
                sessions.append(elements[1].strip())
            else:
                print(f"Incompatible key {elements[0]} in sessions list!")
                raise
        else:
            print(f"Incompatible line {elements} in sessions list!")
            raise

    return sessions

def load_norm_subject_seed(dataname, seedID):
    data = nib.load(dataname).get_fdata(dtype=np.float32)  # individual (time points x space) matrix
    # Because python array starts with 0, where parcel ID starts with 1
    seedID_act = np.array(seedID) - 1
    seedID_act = seedID_act.tolist()
    # - select seed region time-courses
    seeddata = data[:, seedID_act]  # individual (time points x n_seed) matrix
    seedmean = seeddata.mean(axis=1)
    zdata = stats.zscore(seedmean, axis=0)  # - Normalize each time-series
    del data
    return zdata

def load_groupdata_seed(filein, param):
    homedir = filein.sessions_folder
    sublist = filein.sublist
    fname = filein.fname
    gsr = param.gsr
    unit = param.unit
    seedID = param.seedID

    msg = "============================================"
    logging.info(msg)
    msg = "[seed-region] seedID: " + \
        str(seedID) + ", load " + unit + "-level time-series preprocessed with " + gsr + ".."
    logging.info(msg)

    #set up dimensions for subject concatenated array
    tdim = 0
    for idx, subID in enumerate(sublist):
        # - Load fMRI data
        dataname = os.path.join(homedir, str(subID), fname)
        #If data was concatenated using pycap_concatenate, dimensions are saved
        if os.path.exists(dataname + ".npy"):
            dshape = np.load(dataname + ".npy")
        #Otherwise, must load file and get dim directly. Processing inefficent but should be more memory efficient
        else:
            dshape = nib.load(dataname).get_fdata(dtype=np.float32).shape
        tdim += dshape[0]


    seeddata_all = np.empty((tdim, len(sublist)), dtype=np.float32)
    for idx, subID in enumerate(sublist):
        # - Load the mean seed time-course
        dataname = dataname = homedir + str(subID) + "/images/functional/" + fname
        zdata = load_norm_subject_seed(dataname, seedID)
        seeddata_all[:, idx] = zdata

        msg = "(Subject " + str(idx) + ")" + dataname + \
            " : seed average time-series " + str(zdata.shape)
        logging.info(msg)

        del zdata

    msg = ">> Output: a (" + str(seeddata_all.shape[0]) + " x " + \
        str(seeddata_all.shape[1]) + ") array of (seed average time-series x n_subjects)."
    logging.info(msg)
    return seeddata_all

def load_groupdata_seed_usesaved(filein, param):
    # filein.groupdata_seed_filen = filein.datadir + "hpc_groupdata_" + \
    #     param.seedIDname + "_" + param.unit + "_" + param.gsr + "_" + param.spdatatag + ".hdf5"
    filein.groupdata_seed_filen = os.path.join(filein.datadir,  "hpc_groupdata_wb_" + \
        param.seedIDname + "_" + param.unit + "_" + param.gsr + "_" + param.spdatatag + ".hdf5")
    if os.path.exists(filein.groupdata_seed_filen):
        msg = "File exists. Load concatenated seed fMRI/label data file: " + filein.groupdata_seed_filen
        logging.info(msg)

        f = h5py.File(filein.groupdata_seed_filen, 'r')
        seeddata_all = f['seeddata_all']

    else:
        msg = "File does not exist. Load individual seed-region fMRI data."
        logging.info(msg)

        seeddata_all = load_groupdata_seed(filein=filein, param=param)
        f = h5py.File(filein.groupdata_seed_filen, "w")
        dset1 = f.create_dataset(
            "seeddata_all", (seeddata_all.shape[0], seeddata_all.shape[1]), dtype='float32', data=seeddata_all)
        f.close()

        msg = "Saved the average seed fMRI/label data in: " + filein.groupdata_seed_filen
        logging.info(msg)
    return seeddata_all
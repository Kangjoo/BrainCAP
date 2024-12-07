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
from nilearn.masking import apply_mask
import pycap_functions.pycap_exceptions as pe
import h5py
import pandas as pd
import pycap_functions.pycap_utils as utils
# from memory_profiler import profile
# @profile


def load_norm_subject_wb(dataname, mask, bold_type):
    # an individual (time points x space) matrix
    data = nib.load(dataname).get_fdata(dtype=np.float32)

    if mask:
        logging.info("Masking...")
        if bold_type == "CIFTI":
            data = apply_mask_cifti(data, mask)
        else:
            data = apply_mask(data, mask)

    zdata = stats.zscore(data, axis=0)  # Normalize each time-series
    del data
    return zdata



def load_groupdata_wb(filein, param):
    homedir = filein.sessions_folder
    sublist = filein.sublist
    fname = filein.fname
    gsr = param.gsr
    unit = param.unit
    mask_file = param.mask_file
    seed_args = param.seed_args
    if not seed_args:
        seed_based = None
    else:
        seed_based = seed_args['seed_based'].lower() == "yes"
    #seed_based = param.seed_based.lower() == "yes"
    #sdim = param.sdim
    #tdim = param.tdim

    msg = "============================================"
    logging.info(msg)
    msg = "Load " + unit + \
        "-level time-series data preprocessed with " + gsr + ".."
    logging.info(msg)
    
    #set up dimensions for subject concatenated array
    tdim = 0
    sdim = 0
    info_list = []
    for idx, subID in enumerate(sublist):
        dataname = os.path.join(homedir, str(subID), fname)
        #If data was concatenated using pycap_concatenate, dimensions are saved
        if os.path.exists(dataname + ".npy"):
            dshape = np.load(dataname + ".npy")
        #Otherwise, must load file and get dim directly. Processing inefficent but should be more memory efficient
        else:
            dshape = nib.load(dataname).get_fdata(dtype=np.float32).shape
            np.save(dataname + ".npy", dshape) #Save shape so this only has to be done once
        tdim += dshape[0]
        if sdim == 0:
            sdim = dshape[1]
        else:
            if sdim != dshape[1]:
                raise pe.StepError("PyCap Prep - load_groupdata",
                                   f"Different number of features for subject {subID}",
                                   "Compare this subject's data with other subjects")
            
        #Load info data for building and group and labeldata file
        info_path = f"{dataname.split('.')[0]}_info.csv"
        logging.info(info_path)
        if os.path.exists(info_path):
            logging.info(f"session info data found at: {info_path}, loading...")
            info_list.append(pd.read_csv(info_path))

    if info_list != []:
        infodata_filen = os.path.join(filein.datadir, param.tag + param.spdatatag + "_info.csv")
        if os.path.exists(infodata_filen):
            if param.overwrite == "no":
                logging.info("Info data found, overwrite 'no', will not overwrite")
            else:
                logging.info("Info data found, overwrite 'yes', will overwrite")
                pd.concat(info_list).to_csv(infodata_filen, index=False)
        else:
            logging.info("Info data loaded successfuly, saving concatenated file...")
            pd.concat(info_list).to_csv(infodata_filen, index=False)
        del info_list

    if mask_file != None:
        logging.info(f"Mask {mask_file} supplied and will be used")
        #Currently, masks must be an nibabel compatible file
        mask = nib.load(mask_file)
        #Output after masking will be where mask array == 1 or True, so can be used for dimension
        sdim = mask.get_fdata().sum()
    else:
        mask = None

    if seed_based:
        logging.info("Running seed_based analysis")
        seed = seed_args['seed']
        seed_t = utils.get_seedtype(seed)
        seeddata_all = np.empty((tdim, 1), dtype=np.float32)
        # if seed_t == "list": seed_dim = len(seed)
        # elif seed_t == "index": seed_dim = 1
        # #mask
        # elif seed_t == "file":
        #     seed = nib.load(seed)
        #     seed_dim = seed.get_fdata().sum()
        # seeddata_all = np.empty((tdim, seed_dim), dtype=np.float32)
    else:
        seeddata_all = None

    data_all = np.empty((tdim, sdim), dtype=np.float32)
    sublabel_all = np.empty((tdim, ), dtype=np.object_)
    ptr = 0
    for idx, subID in enumerate(sublist):
        # - Load fMRI data
        dataname = os.path.join(homedir, str(subID), fname)
        data = nib.load(dataname).get_fdata(dtype=np.float32)

        if seed_based:
            if seed_t == "file":
                if param.bold_type == "CIFTI":
                    seeddata = apply_mask_cifti(data, seed)
                else:
                    seeddata = apply_mask(data, seed)
            elif seed_t == "index" or seed_t == "list":
                seeddata = data[:, seed]
            #average time-course
            if seed_t != "index":
                seeddata = np.average(seeddata,axis=1)
            seeddata_all[ptr:ptr+seeddata.shape[0], :] = stats.zscore(seeddata, axis=0).reshape((-1,1))
            del seeddata

        if mask:
            logging.info("Masking...")
            if param.bold_type == "CIFTI":
                data = apply_mask_cifti(data, mask)
            else:
                data = apply_mask(data, mask)

        zdata = stats.zscore(data, axis=0)
        #zdata = load_norm_subject_wb(dataname, mask, param.bold_type)
        data_all[ptr:ptr+zdata.shape[0], :] = zdata


        # - Create subject label
        subid_v = [subID] * zdata.shape[0]
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
    return data_all, sublabel_all, seeddata_all

def create_groupdata(filein, param):
    """
    Called by prep function, creates concatenated group data
    """
    filein.groupdata_wb_filen = os.path.join(filein.datadir, param.tag + param.spdatatag + ".hdf5")
    if os.path.exists(filein.groupdata_wb_filen):
        logging.info(f"File {filein.groupdata_wb_filen} exists. Overwrite '{param.overwrite}'")
        if param.overwrite == "no":
            logging.info("Loading...")
            return load_groupdata(filein, param)
        else:
            logging.info("Removing existing data...")
            os.remove(filein.groupdata_wb_filen)

    msg = "Loading individual whole-brain fMRI data."
    logging.info(msg)
    data_all, sublabel_all, seeddata_all = load_groupdata_wb(filein=filein, param=param)
    logging.info("Load success!")
    return data_all, sublabel_all, seeddata_all

def load_groupdata(filein, param):
    filein.groupdata_wb_filen = os.path.join(filein.datadir, param.tag + param.spdatatag + ".hdf5")
    try:
        f = h5py.File(filein.groupdata_wb_filen, 'r')
        data_all = np.array(f['data_all'])
        sublabel_all = utils.index2id(np.array(f['sublabel_all']), filein.sublistfull)
    except Exception as e:
        logging.info(e)
        raise pe.StepError(step="Load concatenated data",
                           error="Cannot load data",
                           action=f"Check prep_pycap outputs: {filein.datadir}")

    return data_all, sublabel_all

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
        #QuNex .bstats loading
        if '.bstats' in motion_data_filen:
            motion_dlist = np.genfromtxt(motion_data_filen, names=True)
            idx = np.where(np.char.find(motion_dlist.dtype.names, motion_type) == 0)
            motion_data_ind = np.genfromtxt(motion_data_filen, skip_header=1, usecols=idx[0])
        #Other motion file loading (csv, tsv, or just line separated single column)
        else:
            if '.tsv' in motion_data_filen:
                sep = '\t'
            else:
                sep = ','
            try:
                if motion_type:
                    m_file = pd.read_csv(motion_data_filen, sep=sep)
                    m_data_ind = m_file[motion_type].to_numpy()
                else:
                    m_file = pd.read_csv(motion_data_filen, header=None, sep=sep)
                    m_data_ind = m_file.to_numpy()
            except:
                raise pe.StepError(step="PyCap prep - frameselection",
                                   error=f'Failed to open motion file {motion_data_filen}',
                                   action="Only compatible with .bstats, .tsv, .csv, or line-seperated single column data." \
                                    "If you have multiple columns, please specify 'motion_type'")

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



# def load_groupdata_wb_daylabel(filein, param):
#     homedir = filein.sessions_folder
#     sublist = filein.sublist
#     fname = filein.fname
#     gsr = param.gsr
#     unit = param.unit
#     mask_file = param.mask

#     msg = "============================================"
#     logging.info(msg)
#     msg = "[whole-brain] Load " + unit + \
#         "-level time-series data preprocessed with " + gsr + ".."
#     logging.info(msg)

#     #set up dimensions for subject concatenated array
#     tdim = 0
#     sdim = 0
#     for idx, subID in enumerate(sublist):
#         # - Load fMRI data
#         dataname = os.path.join(homedir, str(subID), fname)
#         #If data was concatenated using pycap_concatenate, dimensions are saved
#         if os.path.exists(dataname + ".npy"):
#             dshape = np.load(dataname + ".npy")
#         #Otherwise, must load file and get dim directly. Processing inefficent but should be more memory efficient
#         else:
#             dshape = nib.load(dataname).get_fdata(dtype=np.float32).shape
#             np.save(dataname + ".npy", dshape) #Save shape so this only has to be done once
#         tdim += dshape[0]
#         if sdim == 0:
#             sdim = dshape[1]
#         else:
#             if sdim != dshape[1]:
#                 exit() #ERROR

#     if mask_file != None:
#         #What different formats do masks come in?
#         mask = nib.load(mask_file)
#         #Output after masking will be where mask array == 1 or True, so can be used for dimension
#         sdim = mask.get_fdata().sum()
#     else:
#         mask = None

#     data_all = np.empty((len(sublist) * tdim, sdim), dtype=np.float32)
#     sublabel_all = np.empty((len(sublist) * tdim, ), dtype=np.int32)
#     daylabel_all = np.empty((len(sublist) * tdim, ), dtype=np.int32)
#     ptr = 0
#     for idx, subID in enumerate(sublist):
#         # - Load fMRI data
#         dataname = os.path.join(homedir, str(subID), "images", "functional", fname)
#         zdata = load_norm_subject_wb(dataname, mask, param.bold_type)
#         data_all[ptr:ptr+zdata.shape[0], :] = zdata
#         # - Create subject label
#         subid_v = [subID] * zdata.shape[0]
#         subid_v = np.array(subid_v)
#         sublabel_all[ptr:ptr+zdata.shape[0], ] = subid_v
#         # - Creat day label
#         day_v = np.empty(zdata.shape[0]); day_v.fill(1)
#         runlen=int(zdata.shape[0]/2)
#         day_v[runlen:] = 2 
#         daylabel_all[ptr:ptr+zdata.shape[0], ] = day_v
#         # - Update/delete variables
#         ptr += zdata.shape[0]

#         msg = "(Subject " + str(idx) + ")" + dataname + " " + \
#             ", data:" + str(zdata.shape) + ", subject label:" + str(subid_v.shape) + \
#             ", day label:" + str(day_v.shape)
#         logging.info(msg)

#         del zdata, subid_v

#     msg = ">> Output 1: a (" + str(data_all.shape[0]) + " x " + \
#         str(data_all.shape[1]) + ") array of (group concatenated time-series x space)."
#     logging.info(msg)
#     msg = ">> Output 2: a " + str(sublabel_all.shape) + " array of (group concatenated subject label)."
#     logging.info(msg)
#     msg = ">> Output 3: a " + str(daylabel_all.shape[0]) + " array of (group concatenated day label)."
#     logging.info(msg)    
#     return data_all, sublabel_all, daylabel_all

def concatenate_data(files, ndummy, bold_type, bold_labels):
    im_list = []
    conc_labels = []

    if "CIFTI" == bold_type:
        image_header = nib.load(files[0]).header
        im_axis = image_header.get_axis(0)
        for file, label in zip(files, bold_labels):
            fdata = nib.load(file).get_fdata()[ndummy:]
            im_list.append(fdata)
            conc_labels += [label]*fdata.shape[0]
        #images_data = np.vstack([nib.load(file).get_fdata()[ndummy:] for file in files])
        images_data = np.vstack(im_list)
        ax_0 = nib.cifti2.SeriesAxis(start = im_axis.start, step = im_axis.step, size = images_data.shape[0]) 
        ax_1 = image_header.get_axis(1)
        new_h = nib.cifti2.Cifti2Header.from_axes((ax_0, ax_1))
        conc_image = nib.Cifti2Image(images_data, new_h)
        conc_image.update_headers()
    elif "NIFTI" == bold_type:
        for file, label in zip(files, bold_labels):
            fdata = nib.load(file).get_fdata()[:,:,:,ndummy:]
            im_list.append(fdata)
            conc_labels += [label]*fdata.shape[0]
        images_data = np.vstack(im_list)
        #concatenate along time axis
        conc_image = nib.funcs.concat_images(files,axis=3)

    return conc_image, conc_labels

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
    #QuNex sessions list parsing
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

def parse_sfile(sessionsfile):
    groups = None
    #Other parsing
    if '.tsv' in sessionsfile:
        sep = '\t'
    else:
        sep = ','
    s_df = pd.read_csv(sessionsfile, sep=sep)

    if 'session_id' not in s_df.columns:
        raise pe.StepError(step=f"Loading session file {sessionsfile}",
                            error="Missing 'session_id' column",
                            action="Ensure the session file is setup correctly, with session ids specified under 'session_id'")
    sessions = s_df['session_id'].astype(str).to_list()

    if 'group' in s_df.columns:
        groups = s_df['group'].to_list()
    else:
        logging.info("No group labels supplied")
        groups = [None] * len(sessions)
    
    return sessions, groups

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

def apply_mask_cifti(data, mask):
    """
    nilearn apply_mask is not compatible with CIFTI data so must be done manually
    """
    mask_array = mask.get_fdata()
    return data[np.where(mask_array==1)]
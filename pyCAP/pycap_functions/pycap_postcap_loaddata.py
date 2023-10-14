#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Kangjoo Lee (kangjoo.lee@yale.edu)
# Created Date: 04/05/2022
# Last Updated: 04/22/2022
# version ='0.0'
# ---------------------------------------------------------------------------

# =========================================================================
#                   --   Post CAP pipeline template  --
#      Analysis of Co-Activation Patterns(CAP) in fMRI data (HCP-YA)
# =========================================================================

# Prerequisite libraries
#        NiBabel
#              https://nipy.org/nibabel/index.html
#              command: pip install nibabel
# (base) $ conda install matplotlib numpy pandas seaborn scikit-learn ipython h5py memory-profiler kneed nibabel


# Imports
import h5py
import os
import glob
import logging


def load_capoutput(filein, param):

    class TrainingData:
        pass

    training_data = TrainingData()

    class TestData:
        pass

    test_data = TestData()

    # -------------------------------------------
    # - Load CAP output data from training/test datasets
    # -------------------------------------------

    for sp in [1, 2]:

        if sp == 1:
            spdatatag = "training_data"
        elif sp == 2:
            spdatatag = "test_data"
        param.spdatatag = spdatatag

        msg = "============================================"
        logging.info(msg)
        msg = "Start loading " + spdatatag + "..."
        logging.info(msg)

        # Setup data directory
        datadir = filein.datapath + spdatatag + "/"

        # -------------------------------------------
        # - Load CAP output data
        # -------------------------------------------

        clmean_filen = glob.glob(datadir+'*clustermean.hdf5')[0]
        msg = "Load the cluster mean data in " + clmean_filen
        logging.info(msg)
        f = h5py.File(clmean_filen, 'r')
        clmean = f['clmean'][:]
        del f
        msg = "Loaded clmean = " + str(clmean.shape)
        logging.info(msg)

        framecluster_filen = glob.glob(datadir+'*framelabel_clusterID.hdf5')[0]
        msg = "Load the frame-wise clusterID data in " + framecluster_filen
        logging.info(msg)
        f = h5py.File(framecluster_filen, 'r')
        frame_clusterID = f['framecluster'][:]
        del f
        msg = "Loaded frame-wise clusterID = " + str(frame_clusterID.shape)
        logging.info(msg)

        labeldata_fsel_filen = glob.glob(datadir+'Framelabel_subID.hdf5')[0]
        msg = "Load the frame-wise subjectID data in " + labeldata_fsel_filen
        logging.info(msg)
        f = h5py.File(labeldata_fsel_filen, 'r')
        frame_subID = f['labeldata_fsel'][:]
        del f
        msg = "Loaded frame-wise subjectID = " + str(frame_subID.shape)
        logging.info(msg)
        
        daydata_fsel_filen = glob.glob(datadir+'Framelabel_day.hdf5')[0]
        msg = "Load the frame-wise day 1/2 data in " + labeldata_fsel_filen
        logging.info(msg)
        f = h5py.File(daydata_fsel_filen, 'r')
        frame_day = f['daydata_fsel'][:]
        del f
        msg = "Loaded frame-wise day 1/2 info = " + str(frame_day.shape)
        logging.info(msg)        

        if sp == 1:
            training_data.clmean = clmean
            training_data.frame_clusterID = frame_clusterID
            training_data.frame_subID = frame_subID
            training_data.frame_day = frame_day
        elif sp == 2:
            test_data.clmean = clmean
            test_data.frame_clusterID = frame_clusterID
            test_data.frame_subID = frame_subID
            test_data.frame_day = frame_day

    return training_data, test_data

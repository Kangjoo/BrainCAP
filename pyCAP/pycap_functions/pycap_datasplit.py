#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# Created By  : Kangjoo Lee (kangjoo.lee@yale.edu)
# Last Updated: 08/06/2024
# -------------------------------------------------------------------------


# Imports
import math
import h5py
import os
import random
import sklearn.model_selection
import numpy as np
import logging
import pandas as pd


def subsplit(filein, param):
    msg = "============================================"
    logging.info(msg)
    msg = "[Model selection] Population split-half of subjects.."
    logging.info(msg)

    filein.sublistfull = filein.sublist
    splitdata_outfilen = filein.outdir + "subsplit_datalist.hdf5"
    msg = splitdata_outfilen
    logging.info(msg)
    
    if os.path.exists(splitdata_outfilen):

        msg = "File exists. Load existing list of subjects for training/test datasets: " + splitdata_outfilen
        logging.info(msg)

        f = h5py.File(splitdata_outfilen, 'r')
        training_sublist_idx = f['training_sublist_idx']
        test_sublist_idx = f['test_sublist_idx']

        test_sublist = []
        for index in test_sublist_idx:
            test_sublist.append(filein.sublist[index])
        training_sublist = []
        for index in training_sublist_idx:
            training_sublist.append(filein.sublist[index])

        msg = "(Split 1) Training data subjects " + \
            str(len(training_sublist)) + " : " + str(training_sublist)
        logging.info(msg)
        msg = "(Split 2) Test data subjects " + \
            str(len(test_sublist)) + " : " + str(test_sublist)
        logging.info(msg)

    else:
        # generate list of subjects for training/test datasets
        SubIdxlist = np.arange(0, len(filein.sublist))
        spsize = math.floor(SubIdxlist.shape[0]/2)
        if param.subsplittype == "random":
            ms_kwargs = {"train_size": spsize, "test_size": spsize}
            test_sublist_idx, training_sublist_idx = sklearn.model_selection.train_test_split(
                SubIdxlist, **ms_kwargs)

            test_sublist_idx = test_sublist_idx.tolist()
            test_sublist = []
            for index in test_sublist_idx:
                test_sublist.append(filein.sublist[index])
            training_sublist_idx = training_sublist_idx.tolist()
            training_sublist = []
            for index in training_sublist_idx:
                training_sublist.append(filein.sublist[index])

#         elif param.subsplittype == "manual":
#             training_sublist_idx = np.array(range(0, spsize, 1))
#             test_sublist_idx = np.array(range(spsize, spsize*2, 1)) #example method for manual type

        msg = "(Split 1) Training data subjects " + \
            str(len(training_sublist)) + " : " + str(training_sublist)
        logging.info(msg)
        msg = "(Split 2) Test data subjects " + \
            str(len(test_sublist)) + " : " + str(test_sublist)
        logging.info(msg)

        # save output files
        f = h5py.File(splitdata_outfilen, "w")
        dset1 = f.create_dataset("training_sublist_idx", (spsize,),
                                 dtype=int, data=training_sublist_idx)
        dset2 = f.create_dataset("test_sublist_idx", (spsize,), dtype=int, data=test_sublist_idx)
        f.close()
        msg = "Saved Subject split data list indices in " + splitdata_outfilen
        logging.info(msg)

    return test_sublist, training_sublist



def isExistFiles(fileList):
    fexist = []
    for eachFile in fileList:
        if os.path.isfile(eachFile):
            fexist.append(1)
        else:
            fexist.append(0)
    return fexist

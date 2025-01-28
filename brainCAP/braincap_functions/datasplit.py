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

    filein.sublistfull = filein.sublistfull
    #splitdata_outfilen = filein.outdir + "subsplit_datalist.hdf5"
    splitdata_outfilen = os.path.join(filein.datadir, "sessions.hdf5")
    msg = splitdata_outfilen
    logging.info(msg)
    
    if os.path.exists(splitdata_outfilen):

        msg = "File exists. Load existing list of subjects for split_1/split_2 datasets: " + splitdata_outfilen
        logging.info(msg)

        f = h5py.File(splitdata_outfilen, 'r')
        split_1_sublist_idx = f['split_1_sublist_idx']
        split_2_sublist_idx = f['split_2_sublist_idx']

        split_2_sublist = []
        for index in split_2_sublist_idx:
            split_2_sublist.append(filein.sublistfull[index])
        split_1_sublist = []
        for index in split_1_sublist_idx:
            split_1_sublist.append(filein.sublistfull[index])

        msg = "(Split 1) split_1 data subjects " + \
            str(len(split_1_sublist)) + " : " + str(split_1_sublist)
        logging.info(msg)
        msg = "(Split 2) split_2 data subjects " + \
            str(len(split_2_sublist)) + " : " + str(split_2_sublist)
        logging.info(msg)

    else:
        # generate list of subjects for split_1/split_2 datasets
        SubIdxlist = np.arange(0, len(filein.sublistfull))
        spsize = math.floor(SubIdxlist.shape[0]/2)
        #if param.subsplit_type == "random":
        ms_kwargs = {"train_size": spsize, "test_size": spsize}
        split_2_sublist_idx, split_1_sublist_idx = sklearn.model_selection.train_test_split(
            SubIdxlist, **ms_kwargs)

        split_2_sublist_idx = split_2_sublist_idx.tolist()
        split_2_sublist = []
        for index in split_2_sublist_idx:
            split_2_sublist.append(filein.sublistfull[index])
        split_1_sublist_idx = split_1_sublist_idx.tolist()
        split_1_sublist = []
        for index in split_1_sublist_idx:
            split_1_sublist.append(filein.sublistfull[index])

#         elif param.subsplit_type == "manual":
#             split_1_sublist_idx = np.array(range(0, spsize, 1))
#             split_2_sublist_idx = np.array(range(spsize, spsize*2, 1)) #example method for manual type

        msg = "(Split 1) split_1 data subjects " + \
            str(len(split_1_sublist)) + " : " + str(split_1_sublist)
        logging.info(msg)
        msg = "(Split 2) split_2 data subjects " + \
            str(len(split_2_sublist)) + " : " + str(split_2_sublist)
        logging.info(msg)

        # save output files
        f = h5py.File(splitdata_outfilen, "w")
        dset1 = f.create_dataset("split_1_sublist_idx", (spsize,),
                                 dtype=int, data=split_1_sublist_idx)
        dset2 = f.create_dataset("split_2_sublist_idx", (spsize,), dtype=int, data=split_2_sublist_idx)
        f.close()
        msg = "Saved Subject split data list indices in " + splitdata_outfilen
        logging.info(msg)

    return split_2_sublist, split_1_sublist



def isExistFiles(fileList):
    fexist = []
    for eachFile in fileList:
        if os.path.isfile(eachFile):
            fexist.append(1)
        else:
            fexist.append(0)
    return fexist

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# Created By  : Kangjoo Lee (kangjoo.lee@yale.edu)
# Last Updated: 08/06/2024
# -------------------------------------------------------------------------


# Imports
import h5py
import os
import glob
import nibabel as nib
import numpy as np
import argparse
import itertools
import pandas as pd
import logging
import ptitprince as pt
import matplotlib.collections as clt
import matplotlib.pyplot as plt


def corr_maxvarCAP_btw_splits(trainingdata, testdata, param):
    R = np.empty((0, 1))
    for perm in range(param.minperm-1, param.maxperm):
        tmp = np.corrcoef(trainingdata[perm, :], testdata[perm, :])[0, 1]
        tmp = np.array([[tmp]])
        R = np.append(R, tmp, axis=0)
    R = R.reshape(R.shape[0],)
    return R

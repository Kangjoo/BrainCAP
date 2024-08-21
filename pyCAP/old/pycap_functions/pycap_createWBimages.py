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


def createcapimage(clmean, filein):
    msg = "Create " + str(clmean.shape[0]) + \
        " CAP images from input variable: clmean=" + str(clmean.shape)
    logging.info(msg)

    outdir = filein.outdir
    pscalar_filen = filein.pscalar_filen

    for x in range(0, clmean.shape[0]):
        cluster_z = clmean[x, ]
        cluster_z = np.reshape(cluster_z, (1, clmean.shape[1]))
        # - Save the average image within this cluster
        outfilen = outdir + "cluster" + str(x) + "_Zmap.pscalar.nii"
        pscalars = nib.load(pscalar_filen)
        new_img = nib.Cifti2Image(cluster_z, header=pscalars.header,
                                  nifti_header=pscalars.nifti_header)
        new_img.to_filename(outfilen)
        msg = "    cluster " + str(x) + ": saved the average map in " + outfilen
        logging.info(msg)
    return


def generate_meanCAPimg(inputdata, outfilen, filein):
    meanimg = np.mean(inputdata, axis=0)
    meanimg = np.reshape(meanimg, (1, meanimg.shape[0]))
    pscalars = nib.load(filein.pscalar_filen)
    new_img = nib.Cifti2Image(meanimg, header=pscalars.header,
                              nifti_header=pscalars.nifti_header)
    new_img.to_filename(outfilen)
    return

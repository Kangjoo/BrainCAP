#!/usr/bin/env python

import argparse
import os
from pycap_functions.pycap_loaddata import concatenate_data, concatenate_motion, parse_slist
import nibabel as nib
import logging
import time
import numpy as np
import pycap_functions.pycap_exceptions as pe

def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path!")

def file_path(path):
    if os.path.exists(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"file {path} does not exist!")

def local_path(path):
    if path[0] != '/':
        return path
    else:
        raise argparse.ArgumentTypeError(f"{path} must be a local path from the specified sessions_folder!")

parser = argparse.ArgumentParser()
parser.add_argument("-sl", "--sessions_list", required=True, type=file_path, help="Path to list of sessions")
parser.add_argument("-sd", "--sessions_folder", required=True, type=dir_path, help="Data directory containing individual sessions")
parser.add_argument("-b", "--bold_files", type=local_path, required=True, help="Comma separated list of bolds to concatenate, must be inside each session directory")
parser.add_argument("-mf", "--motion_files", type=local_path, required=False, help="Comma separated list of motion files to concatenate if doing scrubbing, must be inside each session directory")
parser.add_argument("-o", "--bold_out", type=local_path, required=True, help="Where to place output concatenated bold files, must be inside each session directory")
parser.add_argument("-om", "--motion_out", type=local_path, required=False, help="Where to place output concatenated motion files, must be inside each session directory. Required for scrubbing")
parser.add_argument("-ow", "--overwrite", type=str, required=False, default="no", help="Whether to overwrite existing files")
parser.add_argument("-d", "--ndummy", type=int, default=0, help="Number of dummy frames to remove")
parser.add_argument("-l", "--log_path", default='./concatenate_bolds.log', help='Path to output log', required=False)

args = parser.parse_args()

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    filename=args.log_path,
                    filemode='w')
console = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

logging.info("PYCAP CONCATENATE BEGIN")

#Wait for a moment so run_pycap.py log tracking can keep up
time.sleep(1)

slist = parse_slist(args.sessions_list)

bold_list = args.bold_files.split(',')
if args.motion_files is not None:
    logging.info("--motion_files supplied, will run motion concatenation")
    conc_motion = True
    motion_list = args.motion_files.split(',')
else:
    logging.info("--motion_files not supplied, skipping motion concatenation")
    conc_motion = False

logging.info(f"Beginning bold concatenation for sessions in {args.sessions_list}...")

for session in slist:
    logging.info(f"    Processing {session}")
    #Check BOLD images are of compatible and same type
    bold_type = None
    for bold in bold_list:
        if 'ptseries' in bold:
            if bold_type == None:
                bold_type = 'CIFTI (parcellated)'
            elif bold_type != 'CIFTI (parcellated)':
                raise pe.StepError(step="Concatenate Bolds", 
                                   error="BOLD type mismatch, for concatenation all BOLDS must be of same type",
                                   action="Check specified BOLDs")
            
        elif 'dtseries' in bold:
            if bold_type == None:
                bold_type = 'CIFTI (dense)'
            elif bold_type != 'CIFTI (dense)':
                raise pe.StepError(step="Concatenate Bolds", 
                                   error="BOLD type mismatch, for concatenation all BOLDS must be of same type",
                                   action="Check specified BOLDs")
            
        elif 'nii.gz' in bold:
            if bold_type == None:
                bold_type = 'NIFTI'
            elif bold_type != 'NIFTI':
                raise pe.StepError(step="Concatenate Bolds", 
                                   error="BOLD type mismatch, for concatenation all BOLDS must be of same type",
                                   action="Check specified BOLDs")
            
        else:
            raise pe.StepError(step="Concatenate Bolds", 
                                error=f"Incompatible file {bold}",
                                action="BOLDs must be of type CIFTI or NIFTI")

    bolds = [os.path.join(args.sessions_folder, session, bold) for bold in bold_list]
    bolds_string = '\t\t\n'.join(bolds)
    logging.info(f"        Searching for files:")
    logging.info(bolds_string)
    logging.info(f"        Running bold concatenation...")
    conc_path = os.path.join(args.sessions_folder, session, args.bold_out)
    if os.path.exists(conc_path):
        logging.info(f"           Warning: Existing concatenated bold file found")
        if args.overwrite.lower() == 'yes':
            logging.info(f"           overwrite=yes, overwriting...")
            conc = concatenate_data(bolds, args.ndummy)
            nib.save(conc, conc_path)
            np.save(conc_path, conc.get_fdata().shape)
            logging.info(f"        File {conc_path} created!")
        else:
            logging.info(f"           overwrite=no, skipping...")
    else:
        conc = concatenate_data(bolds, args.ndummy, bold_type)
        nib.save(conc, conc_path)
        np.save(conc_path, conc.get_fdata().shape)
        logging.info(f"        File {conc_path} created!")

    del conc

    if conc_motion:
        logging.info(f"        Running motion concatenation...")
        motions = [os.path.join(args.sessions_folder, session, motion) for motion in motion_list]
        conc_path = os.path.join(args.sessions_folder, session, args.motion_out)
        if os.path.exists(conc_path):
            logging.info(f"           Warning: Existing concatenated motion file found")
            if args.overwrite.lower() == 'yes':
                logging.info(f"           overwrite=yes, overwriting...")
                with open(conc_path, 'w') as f:
                    f.write('\n'.join(concatenate_motion(motions, args.ndummy)))
                logging.info(f"        File {conc_path} created!")
            else:
                logging.info(f"           overwrite=no, skipping...")
        else:
            with open(conc_path, 'w') as f:
                f.write('\n'.join(concatenate_motion(motions, args.ndummy)))
            logging.info(f"        File {conc_path} created!")

    logging.info(f"        {session} concatenated sucessfully\n")

logging.info(f"--- STEP COMPLETE ---")
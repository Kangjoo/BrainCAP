#!/usr/bin/env python

import argparse
import os
from braincap_functions.loaddata import concatenate_data, concatenate_motion, parse_sfile
import nibabel as nib
import logging
import time
import numpy as np
import pandas as pd
import braincap_functions.exceptions as exceptions
import braincap_functions.utils as utils

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
parser.add_argument( "--sessions_list", required=True, type=file_path, help="Path to list of sessions")
parser.add_argument( "--sessions_folder", required=True, type=dir_path, help="Data directory containing individual sessions")
parser.add_argument("--bold_files", type=local_path, required=True, help="Comma separated list of bolds to concatenate, must be inside each session directory")
parser.add_argument( "--motion_files", type=local_path, required=False, help="Comma separated list of motion files to concatenate if doing scrubbing, must be inside each session directory")
parser.add_argument("--bold_out", type=local_path, required=True, help="Where to place output concatenated bold files, must be inside each session directory")
parser.add_argument( "--motion_out", type=local_path, required=False, help="Where to place output concatenated motion files, must be inside each session directory. Required for scrubbing")
parser.add_argument( "--overwrite", type=str, required=False, default="no", help="Whether to overwrite existing files")
parser.add_argument("--ndummy", type=int, default=0, help="Number of dummy frames to remove")
parser.add_argument("--log_path", default='./concatenate_bolds.log', help='Path to output log', required=False)
parser.add_argument("--bold_type", default=None, help="BOLD data type (CIFTI/NIFTI), if not supplied will use file extention")
parser.add_argument("--bold_labels", default=None, help="BOLD data label, useful for longitudinal analyses")

args = parser.parse_args()

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    filename=args.log_path,
                    filemode='w')
console = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

logging.info("BrainCAP concatenation start")
time.sleep(1)

slist, groups = parse_sfile(args.sessions_list)

#Check BOLD images are of compatible and same type
bold_list = args.bold_files.split('|')

if args.bold_labels is not None:
    bold_labels = args.bold_labels.split('|')
else:
    bold_labels = [None] * len(bold_list)

if bold_labels is not None and len(bold_list) != len(bold_labels):
    raise exceptions.StepError(step='BrainCAP concatenate',
                       error=f'{len(bold_list)} bold files supplied, but {len(bold_labels)} labels given',
                       action="--bold_files and --bold_labels must have the same length")

bold_type = None
for bold in bold_list:
    if not bold_type:
        bold_type = utils.get_bold_type(bold)
    elif bold_type != utils.get_bold_type(bold):
        raise exceptions.StepError()
        
if args.motion_files is not None:
    logging.info("--motion_files supplied, will run motion concatenation")
    conc_motion = True
    motion_list = args.motion_files.split('|')
else:
    logging.info("--motion_files not supplied, skipping motion concatenation")
    conc_motion = False

logging.info(f"Beginning bold concatenation for sessions in {args.sessions_list}...")

for session, group in zip(slist, groups):
    logging.info(f"    Processing {session}")
    logging.info(session)
    logging.info(bold_list)
    bolds = [os.path.join(args.sessions_folder, session, bold) for bold in bold_list]
    bolds_string = '\t\t\n'.join(bolds)
    logging.info(f"        Searching for files:")
    logging.info(bolds_string)
    logging.info(f"        Running bold concatenation...")
    conc_path = os.path.join(args.sessions_folder, session, args.bold_out)
    info_path = os.path.join(args.sessions_folder, session, f"{args.bold_out.split('.')[0]}_info.csv")
    if os.path.exists(conc_path):
        logging.info(f"           Warning: Existing concatenated bold file found")
        if args.overwrite.lower() == 'yes':
            logging.info(f"           overwrite=yes, overwriting...")
            conc, conc_labels = concatenate_data(bolds, args.ndummy, bold_type, bold_labels)
            nib.save(conc, conc_path)
            np.save(conc_path, conc.get_fdata().shape)
            group_list = [group] * len(conc_labels)
            id_list = [session] * len(conc_labels)
            id_list = [session] * len(conc_labels)
            info_df = pd.DataFrame()
            info_df['session_id'] = id_list
            info_df['group'] = group_list
            info_df['label'] = conc_labels
            info_df.to_csv(info_path, index=False)
            logging.info(f"        File {conc_path} created!")
        else:
            logging.info(f"           overwrite=no, skipping...")
    else:
        conc, conc_labels = concatenate_data(bolds, args.ndummy, bold_type, bold_labels)
        nib.save(conc, conc_path)
        np.save(conc_path, conc.get_fdata().shape)
        group_list = [group] * len(conc_labels)
        id_list = [session] * len(conc_labels)
        info_df = pd.DataFrame()
        info_df['session_id'] = id_list
        info_df['group'] = group_list
        info_df['label'] = conc_labels
        info_df.to_csv(info_path, index=False)
        logging.info(f"        File {conc_path} created!")
        del conc, conc_labels, group_list, id_list

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
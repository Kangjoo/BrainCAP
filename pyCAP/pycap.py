#!/usr/bin/env python
#source activate /gpfs/gibbs/pi/n3/software/env/pycap_env

import subprocess
import os
import time
import logging
import copy
from datetime import datetime
from pathlib import Path
import argparse
import pprint
import yaml
import pycap_functions.pycap_exceptions as pe
from pycap_functions.pycap_utils import dict2string

def file_path(path):
    """
    Used for argparse, ensures file exists
    """
    if os.path.exists(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"file {path} does not exist!")
    
import time
def follow(filepath, threshold=None):
    """
    Will print the text of a file as it is being written

    If threshold is supplied, will stop (threshold * 0.1ms) after the file has
    received no new text. Otherwise, will continue until no longer called.
    """
    file = open(filepath, "r")
    file.seek(0,2)
    stall_counter = 0
    while True:
        line = file.readline()
        if not line:
            if threshold is not None and stall_counter > threshold:
                break
            stall_counter += 1
            time.sleep(0.01)
            continue
        stall_counter = 0
        yield line

def convert_bool(arg_dict):
    """
    Convert 'True' and 'False' to 'yes' and 'no'

    YAML accepts bool (and will even convert string to bools). For parsing CLI arguments
    we need strings.
    """
    for arg in arg_dict.keys():
        if isinstance(arg_dict[arg], dict):
            arg_dict[arg] = convert_bool(arg_dict[arg])
        if isinstance(arg_dict[arg], bool):
            if arg_dict[arg]: arg_dict[arg] = 'yes'
            else: arg_dict[arg] = 'no'

    return arg_dict
    
#delimiter for lists in CLI strings
list_delim = '|'

#Main function

timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
pp = pprint.PrettyPrinter(indent=1)

# #Dict containing required and optional parameters for each step, should exactly match the actual flags
# #includes internally set params (like log_path and permutation)
arg_dict = {'required':
                {'concatenate_bolds':
                 ['sessions_list', 'sessions_folder', 'bold_files', 'bold_out', 'log_path'],
                 'prep':
                 ['sessions_list','permutations','gsr','sessions_folder','bold_path','analysis_folder','log_path'],
                 'clustering':
                 ['sessions_list','sessions_folder','analysis_folder','log_path', 'cluster_args'],
                 'post':
                 ['sessions_list','permutations','sessions_folder','analysis_folder','log_path', 'cluster_args']
                 }, 
            'optional':
                {'concatenate_bolds':
                 ['overwrite', 'ndummy', 'motion_files', 'motion_out','bold_type', 'bold_labels'],
                 'prep':
                 ['tag','scrubbing','motion_type','motion_path','seed_args','motion_threshold','overwrite', 'event_combine','event_type','bold_type','display_motion'],
                 'clustering':
                 ['tag','overwrite','permutation','bold_type'],
                 'post':
                 ['tag','scrubbing','motion_type','motion_path','motion_threshold','overwrite', 'event_combine','event_type', 'save_image', 'parc_file','bold_type', 'save_stats', 'cluster_selection', 'use_all']
                 }
            }

#Dict containing script paths for each step
script_path = os.path.dirname(os.path.realpath(__file__))
step_dict = {'concatenate_bolds':f'{script_path}/pycap_concatenate.py',
             'prep':f'{script_path}/pycap_prep.py',
             'clustering':f'{script_path}/pycap_clustering.py',
             'post':f'{script_path}/pycap_post.py'}

schedulers = ['NONE','SLURM']

defaults = {"global":{'scheduler':{'type':"NONE"}, 'analysis_folder':'.'},
            "concatenate_bolds":{'overwrite':'no'}}


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=file_path, required=True, help="Path to config file with PyCap parameters")
parser.add_argument("--steps", type=str, help="Comma seperated list of PyCap steps to run. Can also be specified in config")
parser.add_argument("--dryrun", type=str, default="no", help="Dry-Run which will not actually launch steps")
args, unknown = parser.parse_known_args()
args = vars(args)

with open(args['config'], 'r') as f:
    config = yaml.safe_load(f)

#Combine command line args with config args, with command line overwriting
args = config | args

if 'steps' not in args.keys():
    print("--steps parameter not supplied! Exiting...")
    exit()

if type(args['steps']) != list:
    args['steps'] = args['steps'].split(',')

#Set global defaults
args['global'] = defaults['global'] | args['global'] 
parsed_args = {}
parsed_args['global'] = args['global']
#Check schedulers are the same and parse with defaults
first=True
for step in args['steps']:
    if step not in step_dict.keys():
        print(f"--steps parameter: '{step}' invalid! Valid steps: {step_dict.keys()}")
        exit()

    #step_args = defaults[step] | args['global'] | args[step] #NEED TO CHANGE
    step_args = args['global'] | args[step]
    if 'type' in step_args['scheduler'].keys():
        step_sched = step_args['scheduler']['type'].upper()
        if step_sched not in schedulers:
            print(f"ERROR: Invalid scheduler {step_sched} provided! Scheduler must be one of '{str(schedulers)}'. Exiting...")
            exit()
    else:
        print(f"ERROR: scheduler specified in config, but missing scheduler type!")
        exit()

    if first: 
        sched_type = step_sched
        first=False
    elif sched_type != step_sched:
        print(f"ERROR: all specified schedulers must be of same type, expected {sched_type}, found {step_sched}!")
        print(f"If scheduler type unspecified, assumed 'NONE'. Exiting...")
        exit()

    parsed_args[step] = step_args

#Command orchestration

#Contain list of commands to run, either as jobs or directly
commands = []
#Contains list of output logs (command logs for no scheduler, job logs for scheduler)
logs = []
#used for job dependencies
steps = []

#Setup commands and params for each step
for step in args['steps']:

    step_args = convert_bool(copy.deepcopy(parsed_args[step]))

    if 'logs_folder' not in step_args.keys():
        step_args['logs_folder'] = os.path.join(step_args['analysis_folder'], 'logs')
    if not os.path.exists(step_args['logs_folder']):
        os.makedirs(step_args['logs_folder'])


    var_key=None
    if 'cluster_args' in step_args.keys():
        if '_variable' in step_args['cluster_args'].keys(): var_key = step_args['cluster_args']['_variable']
        for ckey, cval in step_args['cluster_args'].items():
            if isinstance(cval, list):
                if not var_key: 
                    var_key = ckey
                elif var_key != ckey:
                    raise pe.StepError(step="PyCap Clustering Orchestration",
                                        error="Only one variable can be defined as a list for parallelization!",
                                        action=f"Check cluster_args keys, error caused by {var_key} and {ckey}")

    #compatible steps can make use of ntasks and jobs
    if step == "clustering":
        #Find the clusterig variable which will be defined as a list (or not at all)
        #Used for parallelization, so can only be one such variable
        
        if not var_key:
            print("Only single parameter values supplied, not running within-permutation parallelization")        
            step_args['_cvar'] = None
            step_args['_cvarlist'] = None
            ntasks = 1

        else:
            step_args['_cvar'] = var_key
            step_args['cluster_args']['_variable'] = var_key
            step_args['_cvarlist'] = step_args['cluster_args'].pop(var_key)
            ntasks = len(step_args['_cvarlist'])
        jobs = step_args.pop('permutations')
        #del step_args['permutations']
    else:
        ntasks = 1
        jobs = 1
        min_k = None

    #job level, used for permutations
    for job in range(jobs):
        split = job+1
        #Setup log path
        #Running as job directs output to it's own log, in which case better to use that for monitoring
        #Otherwise, commands and logs are launched at the 'task' level below
        if jobs == 1:
            step_args['log_path'] = os.path.join(step_args['logs_folder'], f'PyCap_{step}_{timestamp}.log')
            if sched_type != "NONE":
                sched_log = os.path.join(step_args['logs_folder'], f'PyCap_{sched_type}_{step}_{timestamp}.log')
                logs.append(sched_log)
                job_name = f"{step}"
        else:
            step_args['log_path'] = os.path.join(step_args['logs_folder'], f'PyCap_{step}_perm{split}_{timestamp}.log')
            if sched_type != "NONE":
                sched_log = os.path.join(step_args['logs_folder'], f'PyCap_{sched_type}_{step}_perm{split}_{timestamp}.log')
                logs.append(sched_log)
                job_name = f"{step}_perm{split}"


        # if sched_type != "NONE":
        #     sched_log = os.path.join(step_args['logs_folder'], f'PyCap_{sched_type}_{step}_perm{split}_{timestamp}.log')
        #     logs.append(sched_log)
        
        pp.pprint(step_args)

        #Command headers
        if sched_type.upper() == "NONE":
            command = ""

        elif sched_type.upper() == "SLURM":
            command = "#!/bin/bash\n"
            command += f"#SBATCH -J {job_name}\n"
            command += f"#SBATCH --mem-per-cpu={step_args['scheduler']['cpu_mem']}\n"
            command += f"#SBATCH --cpus-per-task={step_args['scheduler']['cpus']}\n"
            command += f"#SBATCH --partition={step_args['scheduler']['partition']}\n"
            command += f"#SBATCH --time={step_args['scheduler']['time']}\n"
            command += f"#SBATCH --output={sched_log}\n"
            if "account" in step_args['scheduler'].keys():
                command += f"#SBATCH --account={step_args['scheduler']['account']}\n"
            command += f"#SBATCH --nodes=1\n"
            command += f"#SBATCH --ntasks=1\n"
            

        elif sched_type.upper() == "PBS":
            pass

        if ntasks != 1 and sched_type.upper() != "NONE":
            task = "${PERM}"
            step_args['log_path'] = os.path.join(step_args['logs_folder'], f'PyCap_{step}_perm{split}_{task}_{timestamp}.log')

        #parameters that depend on task
        if step == "clustering":
            step_args['permutation'] = split
            if sched_type.upper() != "NONE":
                command += f"#SBATCH --array={','.join(map(str,step_args['_cvarlist']))}\n"
                command += "PERM=${SLURM_ARRAY_TASK_ID} \n"
                step_args['cluster_args'][step_args['_cvar']] = "$PERM"
            else:
                step_args['cluster_args'][step_args['_cvar']] = step_args['_cvarlist']



        command += f"python {step_dict[step]} "

        for arg in arg_dict['required'][step]:
            if arg not in step_args.keys():
                print(f"ERROR! Missing required argument '{arg}' for PyCap '{step}'. Exiting...")
                exit()
            if type(step_args[arg]) == list:
                command += f'--{arg} "{list_delim.join(map(str,step_args[arg]))}" ' 
            elif type(step_args[arg]) == dict:
                command += f'--{arg} {dict2string(step_args[arg])} ' 
            else:
                command += f'--{arg} "{step_args[arg]}" '

        for arg in arg_dict['optional'][step]:
            if arg in step_args.keys():
                if type(step_args[arg]) == list:
                    command += f'--{arg} "{list_delim.join(map(str,step_args[arg]))}" ' 
                elif type(step_args[arg]) == dict:
                    command += f'--{arg} {dict2string(step_args[arg])} ' 
                else:
                    command += f'--{arg} "{step_args[arg]}" '

        #commands launched at task level
        if sched_type == "NONE":
            commands.append(command)
            logs.append(step_args['log_path'])
            steps.append(step)
            command = ""

        else:
            commands.append(command)
            steps.append(step)
            

prev_ids = None
new_ids = []
prev_step = None
#Run commands
for command, log, step in zip(commands, logs, steps):
    serr = subprocess.STDOUT
    sout = subprocess.PIPE
    if sched_type.upper() == "NONE":
        if args['dryrun'] == "yes":
            print(command)
            continue

        run = subprocess.Popen(
                command, shell=True, stdin=subprocess.PIPE, stdout=sout, stderr=serr, close_fds=True)
        
        #Wait a moment so that the file is generated
        t = 0
        print(f"Running command:\n {command}\n")
        print(f"Output log: {log}")
        while not os.path.exists(log):
            if t > 5000:
                print("ERROR! Step failed to launch, halting execution!")
                print(f"Attempted to run the following command:\n {command}")
                exit()
            time.sleep(0.01)
            t += 1
        print(f"Command launched succesfully! Showing output from {log}")
        #follows step output and prints it
        runlog = follow(log, 6000)
        for line in runlog:
            p_line = line.replace("\n","")
            print(p_line)

            if "STEP COMPLETE" in p_line:
                print("Step completed successfully!")
                break

            if "STEP FAIL" in p_line:
                print("ERROR! Step failed, halting execution!")
                break

    elif sched_type.upper() == "SLURM":
        
        #If new step, swap dependencies
        if step != prev_step:
            prev_ids = ":".join(new_ids)
            new_ids = []

        if prev_ids != '':
            run_com = f"sbatch --dependency afterok:{prev_ids}"
        else:
            run_com = "sbatch"

        if args['dryrun'] == "yes":
            print(run_com)
            print(command)
            continue

        run = subprocess.Popen(
                run_com, shell=True, stdin=subprocess.PIPE, stdout=sout, stderr=serr, close_fds=True
            )
        run.stdin.write((command).encode("utf-8"))
        out = run.communicate()[0].decode("utf-8")
        run.stdin.close()

        new_ids.append(out.split("Submitted batch job ")[1].replace('\n',''))
        prev_step = step
        print(f"Launched job {new_ids[-1]}")
        print("Follow command progress in:")
        print(f"{log}")

print("All steps launched!")
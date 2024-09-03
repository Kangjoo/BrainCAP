#!/usr/bin/env python
#source activate /gpfs/gibbs/pi/n3/software/env/pycap_env

import subprocess
import os
import time
import logging
from datetime import datetime
from pathlib import Path
import argparse
import pprint
import yaml

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
            time.sleep(0.1)
            continue
        stall_counter = 0
        yield line
    
#Main function

timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
pp = pprint.PrettyPrinter(indent=1)

#Dict containing required and optional parameters for each step, should exactly match the actual flags
#includes internally set params (like log_path)
arg_dict = {'required':
                {'concatenate_bolds':
                 ['sessions_list', 'sessions_folder', 'bold_files', 'bold_out', 'log_path'],
                 'prep':
                 ['sessions_list','gsr','sessions_folder','bold_path','analysis_folder']
                 }, 
            'optional':
                {'concatenate_bolds':
                 ['overwrite', 'ndummy', 'motion_files', 'motion_out', 'log_path'],
                 'prep':
                 ['n_splits','scrubbing','motion_type','motion_path','seed_type','seed_name','seed_threshtype','seed_threshold','subsplit_type','time_threshold','motion_threshold','display_motion','overwrite']
                 }
            }

#Dict containing script paths for each step
step_dict = {'concatenate_bolds':'/gpfs/gibbs/pi/n3/Studies/CAP_Time_Analytics/time-analytics/pyCAP/pyCAP/pycap_concatenate.py',
             'prep':'/gpfs/gibbs/pi/n3/Studies/CAP_Time_Analytics/time-analytics/pyCAP/pyCAP/pycap_prep.py'}

schedulers = ['NONE','SLURM','PBS']

defaults = {"global":{'scheduler':{'type':"NONE"}, 'analysis_folder':'.'},
            "concatenate_bolds":{'overwrite':'no'}}


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=file_path, required=True, help="Path to config file with PyCap parameters")
parser.add_argument("--steps", type=str, help="Comma seperated list of PyCap steps to run. Can also be specified in config")
parser.add_argument("--debug", type=str, default="no", help="Debug")
args = vars(parser.parse_args())

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

#Setup commands and params for each step
for step in args['steps']:

    step_args = parsed_args[step]
    #Setup log path
    if 'logs_folder' not in step_args.keys():
        step_args['logs_folder'] = os.path.join(step_args['analysis_folder'], 'logs')
    if not os.path.exists(step_args['logs_folder']):
        os.makedirs(step_args['logs_folder'])
    step_args['log_path'] = os.path.join(step_args['logs_folder'], f'{step}_{timestamp}.log')
    #Running as job directs output to it's own log, in which case better to use that for monitoring
    if sched_type != "NONE":
        sched_log = os.path.join(step_args['logs_folder'], f'{sched_type}_{step}_{timestamp}.log')
        logs.append(sched_log)
    else:
        logs.append(step_args['log_path'])
    
    pp.pprint(step_args)

    #Build step command
    command = ""

    #Currently unused
    if sched_type.upper() == "NONE":
        pass

    elif sched_type.upper() == "SLURM":
        command = "#!/bin/bash\n"
        command += f"#SBATCH -J {step}\n"
        command += f"#SBATCH --mem-per-cpu={step_args['scheduler']['cpu_mem']}\n"
        command += f"#SBATCH --cpus-per-task={step_args['scheduler']['cpus']}\n"
        command += f"#SBATCH --partition={step_args['scheduler']['partition']}\n"
        command += f"#SBATCH --time={step_args['scheduler']['time']}\n"
        command += f"#SBATCH --output={sched_log}\n"
        command += f"#SBATCH --nodes=1\n"
        command += f"#SBATCH --ntasks=1\n"

        #SLURM output
        logs.append(f"{step}_{timestamp}")

    elif sched_type.upper() == "PBS":
        pass
    
    command += f"python {step_dict[step]} "

    #Required Params
    for arg in arg_dict['required'][step]:
        if arg not in step_args.keys():
            print(f"ERROR! Missing required argument '{arg}' for {step}. Exiting...")
            exit()
        if type(step_args[arg]) == list:
            step_args[arg] = ','.join(step_args[arg])
        command += f"--{arg} {step_args[arg]} "

    for arg in arg_dict['optional'][step]:
        if arg in step_args.keys():
            if type(step_args[arg]) == list:
                step_args[arg] = ','.join(step_args[arg])
            command += f"--{arg} {step_args[arg]} "

    commands.append(command)

job_id = None
#Run commands
for command, log in zip(commands, logs):
    serr = subprocess.STDOUT
    sout = subprocess.PIPE
    if sched_type.upper() == "NONE":
        if args['debug'] == "yes":
            print(command)
        else:
            run = subprocess.Popen(
                    command, shell=True, stdin=subprocess.PIPE, stdout=sout, stderr=serr, close_fds=True)
            
            #Wait a moment so that the file is generated
            t = 0
            while not os.path.exists(log):
                if t > 1000:
                    print("ERROR! Step failed to launch, halting execution!")
                    print(f"Attempted to run the following command:\n {command}")
                    exit()
                time.sleep(0.1)
                t += 1
            print(f"Running command:\n {command}\n")
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
        
        #Set previous job as a dependancy
        if job_id != None:
            run_com = f"sbatch --dependency afterok:{job_id}"
        else:
            run_com = "sbatch"

        run = subprocess.Popen(
                run_com, shell=True, stdin=subprocess.PIPE, stdout=sout, stderr=serr, close_fds=True
            )
        run.stdin.write((command).encode("utf-8"))
        out = run.communicate()[0].decode("utf-8")
        run.stdin.close()

        job_id = out.split("Submitted batch job ")[1]
        print(f"Launched job {job_id}")
        print("Follow command progress in:")
        print(f"{log}")

print("All steps launched!")
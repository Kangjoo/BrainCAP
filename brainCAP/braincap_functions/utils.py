import logging
import braincap_functions.exceptions as pe
import numpy as np
import os

def handle_args(args, req_args, step=None, param=None):
    bad_arg = _check_args(args, req_args)
    if bad_arg != []:
        raise pe.MissingParameterError(step, param, bad_arg)
    return

def _check_args(args, req_args):
    """
    Check whether some namespace arguments (parser) are None or missing and return them
    """
    bad_args = []
    args_dict = vars(args)
    for arg in req_args:
        try:
            if args_dict[arg] == None:
                bad_args.append(arg)
        except:
            bad_args.append(arg)
    return bad_args

def get_bold_type(bold):
    """
    Checks whether a bold is CIFTI or NIFTI
    """
    if "ptseries" in bold or "dtseries" in bold:
        return "CIFTI"
    elif "nii" in bold:
        return "NIFTI"
    else:
        return None
    
def dict2string(to_convert):
    """
    Converts a dict to a CLI friendly string
    """
    #uses " for CLI parsing
    out = '"'
    for key,val in to_convert.items():
        #list denoted by pipe
        if isinstance(val, list): val = '|'.join(map(str,val))
        out += f'{key}={val},'
    #Remove unneccesary comma
    out = out[:-1]
    out += '"'
    return out

def string2dict(to_convert):
    """
    Converts a (usually) CLI string to dict
    """
    #In case using dict2string but not parsed through CLI for some reason
    to_convert = to_convert.strip('"')
    out = {}
    for pair in to_convert.split(','):
        try:
            key,val = pair.split("=")
        except:
            raise pe.StepError(step="CLI parsing",
                               error=f"Unable to parse due to unpaired '=' character in {to_convert}!",
                               action="The character '=' must denote a key-value pair!")
        #Converting potential non-string values
        if '|' in val: 
            val = val.split('|')
            for i in range(len(val)):
                val[i] = _string2type(val[i])
        else:
            val = _string2type(val)
        out[key] = val
    return out

def _string2type(val):
    """
    Check if string appears to be int, float, or bool and convert it if so
    """
    if val.isdecimal(): val = int(val)
    elif '.' in val: 
        if val.replace('.','',1).isdecimal(): val = float(val)
    elif val == 'True': val = True
    elif val == 'False': val = False
    return val

def id2index(to_convert, sublistfull):
    """
    Convert subject ids to index of sublist. Used for h5py storage
    """
    converted = np.zeros(len(to_convert))
    sublist = np.asarray(sublistfull, dtype=str)
    to_convert = np.asarray(to_convert, dtype=str)
    for i in range(len(converted)):
        index = np.where(to_convert[i]==sublist)[0]
        # if len(index) != 1:
        #     raise pe.StepError(step="Session list parsing",
        #                        error=f"Found session {to_convert[i]} {len(index)} times in sessions list!",
        #                        action="Session list and loaded data may not match, check your parameters")
        converted[i] = index[0]
    return converted

def index2id(to_convert, sublistfull):
    """
    Convert sublist index to subject id
    """
    sublist = np.asarray(sublistfull)
    return sublist[to_convert]

def get_seedtype(seed):
    """
    Checks whether seed is a mask file or list of indices and checks incompatibilites
    """
    if isinstance(seed, list):
        for val in seed:
            seed_t = get_seedtype(val)
            if seed_t != "index":
                raise pe.StepError(step="loading seed",
                                   error=f"seed list only compatible with type 'index', but found {seed_t}",
                                   action="check parameters and documentation")
        seed_t = "list"
            
    elif isinstance(seed, int):
        seed_t = "index"

    elif isinstance(seed, str):
        if os.path.exists(seed):
            seed_t = "file"
        else:
            raise pe.StepError(step="loading seed",
                            error=f"attempted to load {seed} as a file but it does not exist",
                           action="Check the supplied path exists and is not a local path.")
    
    else:
        raise pe.StepError(step="loading seed",
                           error=f"incompatible type for {seed}",
                           action="seed must be a file, index, or list of indices")

    return seed_t
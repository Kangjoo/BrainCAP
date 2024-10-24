import logging
import pycap_functions.pycap_exceptions as pe

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
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

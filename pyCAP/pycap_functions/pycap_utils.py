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

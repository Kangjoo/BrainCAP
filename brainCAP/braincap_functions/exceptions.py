import logging

class MissingParameterError(Exception):
    """There was a parameter missing when running the step."""
    
    def __init__(self, step=None, param=None, req_param=None):
        if step is None:
            step = "UNKNOWN"
        if req_param is None:
            req_param = "UNSPECIFIED"

        msg = f"ERROR: Step '{step}' has failed due to missing parameter(s) '{req_param}'"
        
        if param is not None:
            msg += f", which is required for '{param}'"

        super(MissingParameterError, self).__init__(msg)
        logging.info(msg)
        logging.info("--- STEP FAIL ---")

class StepError(Exception):
    """There was an error when running the step."""
    def __init__(self, step=None, error=None, action=None):
        if step is None:
            step = "UNKNOWN"
        if error is None:
            error = "UNSPECIFIED"

        msg = f"ERROR: Step '{step}' has failed due to {error}. "

        if action is not None:
            msg += f"Recommended action: {action}"

        super(StepError, self).__init__(msg)
        logging.info(msg)
        logging.info("--- STEP FAIL ---")

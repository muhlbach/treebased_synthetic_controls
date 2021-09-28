#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
import numpy as np
from collections.abc import Sequence
#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------
def check_estimator(estimator):
    """Check whether model instance is useful for estimating propensity scores"""
    # Methods required
    required_methods = ["fit", "predict","get_params", "set_params"]

    # Check if class has required methods by using getattr() to get the attribute, and callable() to verify it is a method
    for method in required_methods:
        method_attribute = getattr(estimator, method, None)
        if not callable(method_attribute):
            raise Exception(f"Estimator '{type(estimator).__name__}' does not contain the method '{method}'")

def check_param_grid(param_grid):
    if hasattr(param_grid, "items"):
        param_grid = [param_grid]

    for p in param_grid:
        for name, v in p.items():
            if isinstance(v, np.ndarray) and v.ndim > 1:
                raise ValueError("Parameter array should be one-dimensional.")

            if isinstance(v, str) or not isinstance(v, (np.ndarray, Sequence)):
                raise ValueError(
                    "Parameter grid for parameter ({0}) needs to"
                    " be a list or numpy array, but got ({1})."
                    " Single values need to be wrapped in a list"
                    " with one element.".format(name, type(v))
                )

            if len(v) == 0:
                raise ValueError(
                    "Parameter values for parameter ({0}) need "
                    "to be a non-empty sequence.".format(name)
                )
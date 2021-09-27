#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

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


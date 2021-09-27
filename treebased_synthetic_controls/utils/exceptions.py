#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
import re

#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------
# class CateError(Exception):
#     """
#     Exception raised when heterogeneous treatment effects are not available
#     """
#     def __init__(self, message="Heterogeneous treatment effects are not available."):
#         self.message = message
#         super().__init__(self.message)    

# class DfError(Exception):
#     """
#     Exception raised when input is not a pd.DataFrame
#     """
#     def __init__(self, input, message=f"Input must be an instance of 'pd.DataFrame' but is currently 'INPUT_TYPE'"):
#         self.input = input
#         self.message = message
#         super().__init__(self.message)    

#     def __str__(self):
#         return re.sub("INPUT_TYPE", type(self.input).__name__, self.message)
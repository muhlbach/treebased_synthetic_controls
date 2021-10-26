#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
import numpy as np
import pandas as pd
import math

# User
from .exceptions import WrongInputException
from .tools import break_links
#------------------------------------------------------------------------------
# Non-balanced IID bootstrap
#------------------------------------------------------------------------------
class Bootstrap(object):
    """
    Various bootstraps
    """
    # --------------------
    # Constructor function
    # --------------------
    def __init__(self,
                 bootstrap_type="iid",
                 n_bootstrap_samples=100,
                 block_length=5,
                 ):
        # Initialize inputs
        self.bootstrap_type = bootstrap_type
        self.n_bootstrap_samples = n_bootstrap_samples
        self.block_length = block_length

    # --------------------
    # Class variables
    # --------------------
    BOOTSTRAP_TYPE_ALLOWED = ["iid", "circular"]
    X_ALLOWED = ["pd.Series", "pd.DataFrame", "np.ndarray"]
    
    # --------------------
    # Private functions
    # --------------------

    # --------------------
    # Public functions
    # --------------------
    def generate_samples(self, x):

        x = break_links(x)        
        
        if isinstance(x, pd.Series):
            pass
        elif isinstance(x, pd.DataFrame):
            x = x.squeeze()
        elif isinstance(x, np.ndarray):
            x = pd.Series(x)
        else:
            raise WrongInputException(input_name="x", provided_input=x, allowed_inputs=self.X_ALLOWED)
        
        # Get size
        n_obs = x.shape[0]
        
        # Save original indices
        original_indices = x.index
        
        # Reset index
        x.reset_index(drop=True,inplace=True)
        
        if self.bootstrap_type=="iid":        
            # Generate random integer indices from 0 to n-1
            indices = np.random.randint(low=0, high=n_obs, size=(n_obs, self.n_bootstrap_samples))
            
        elif self.bootstrap_type=="circular":
            # Stack observations such that all blocks have the same option of being chosen
            x = pd.concat([x, x.iloc[0:(self.block_length-1)]], axis=0).reset_index(drop=True)
            
            # 
            block_start_indices = np.array(range(n_obs))            
            add_indices  = np.arange(self.block_length, dtype=int).reshape((1, self.block_length))
            
            indices = np.random.choice(block_start_indices,
                                       size=(self.n_bootstrap_samples,math.ceil(n_obs / self.block_length),1),
                                       replace=True)
            
            # Add successive indices
            indices = (indices + add_indices)

            # Reshape
            indices = indices.ravel().reshape((n_obs,-1), order="F")
        
        # Sample
        x_resampled = pd.DataFrame(np.array(x)[indices])

        # Re-index
        x_resampled.index = original_indices
        
        return x_resampled

    
        
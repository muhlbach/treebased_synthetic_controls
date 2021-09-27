#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# User


#------------------------------------------------------------------------------
# Empty class
#------------------------------------------------------------------------------
class SyntheticControl(object):
    """
    This class estimates the average treatment effects by constructing a synthetic control group.
    The construction of the synthetic control group is governed by the supplied regressor and it could be any regressor
    """
    
    
    # --------------------
    # Constructor function
    # --------------------
    def __init__(self,
                 estimator=RandomForestRegressor(),
                 params=None,
                 cv_folds=5,
                 fold_type="block",
                 
                 verbose=False,
                 ):
        # Initialize inputs
        self.estimator = estimator
        self.verbose = verbose

    # --------------------
    # Class variables
    # --------------------

    # --------------------
    # Private functions
    # --------------------

    # --------------------
    # Public functions
    # --------------------
    def fit(self,Y,W,X):
        
        # Find masks for the different periods pre- and post-treatment
        self.idx_T0 = W == 0
        self.idx_T1 = W == 1
        
        ## Split data into pre-treatment and post-treatment
        # Pre-treatment (Y0,X)
        Y_pre, X_pre = Y.loc[self.idx_T0], X.loc[self.idx_T0,:]
        
        # Post-treatment (Y1,X)
        Y_post, X_post = Y.loc[self.idx_T1], X.loc[self.idx_T1,:]
                
        # Estimate f in Y0 = f(X) + eps
        self.estimator.fit(X=X_pre,y=Y_pre)
                
        # Predict Y0 post-treatment
        Y_post_hat = self.estimator.predict(X=X_post)
        
        # Get descriptive statistics of Y both pre- and post-treatment
        self.Y_pre_mean = Y_pre.mean()
        self.Y_post_mean = Y_post.mean()
        self.Y_post_hat_mean = Y_post_hat.mean()
        
    
    def calculate_average_treatment_effect(self, X_post_treatment=None):
        
        if X_post_treatment is None:
            Y_post_hat_mean = self.Y_post_hat_mean
        else:
            # Predict Y0 post-treatment
            Y_post_hat = self.estimator.predict(X=X_post_treatment)
            
            # Average
            Y_post_hat_mean = Y_post_hat.mean()
            
        # Difference between Y1 and Y0-predicted
        Y_diff = self.Y_post_mean - Y_post_hat_mean
        
        # Estimated treatment effect
        tau_ave = Y_diff.mean()        

        return tau_ave
            
        
        

        
        
        
        
        
    
        
    
    
    
    
    
    
    
    
    
    
        
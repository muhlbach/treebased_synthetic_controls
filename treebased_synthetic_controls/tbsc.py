#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold, TimeSeriesSplit

# User
from utils.sanity_check import check_param_grid
from utils.tools import SingleSplit
from utils.exceptions import WrongInputException
from utils.bootstrap import Bootstrap

#------------------------------------------------------------------------------
# Synthetic Control Group Method
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
                 estimator=RandomForestRegressor(criterion='mse',
                                                 min_weight_fraction_leaf=0.0,
                                                 min_impurity_decrease=0.0,
                                                 bootstrap=False,
                                                 oob_score=False,
                                                 n_jobs=None,
                                                 random_state=None,
                                                 verbose=0,
                                                 warm_start=False,
                                                 ccp_alpha=0.0,
                                                 max_samples=None),
                 param_grid={'n_estimators': 500,
                             'max_depth': [2,4,8,16,None],
                             'min_samples_split': [2,4,8,16],
                             'min_samples_leaf': [1,2,4,8],
                             'max_features': [1/4,1/3,1/2,2/3, 'sqrt','log2'],
                             'max_leaf_nodes': None,
                             },
                 cv_params={'scoring':None,
                            'n_jobs':None,
                            'refit':True,
                            'verbose':0,
                            'pre_dispatch':'2*n_jobs',
                            'random_state':None,
                            'error_score':np.nan,
                            'return_train_score':False},
                 n_folds=3,
                 fold_type="KFold",
                 max_n_models=50,
                 test_size=0.25,
                 verbose=False,
                 ):
        # Initialize inputs
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv_params = cv_params
        self.n_folds = n_folds
        self.fold_type = fold_type
        self.max_n_models = max_n_models
        self.test_size = test_size
        self.verbose = verbose

        # Set param_grid values to list if not already list
        self.param_grid = {k: list(set(v)) if isinstance(v, list) else v.tolist() if isinstance(v, np.ndarray) else [v] for k, v in self.param_grid.items()}

        # Compute number of models
        self.n_models = np.prod(np.array([len(v) for k,v in self.param_grid.items()]))

        # Define data splitter, which depends on the number of folds.
        if self.n_folds==1:
            self.splitter = SingleSplit(test_size=self.test_size)
            
        elif self.n_folds>=2:
            if self.fold_type=="KFold":
                self.splitter = KFold(n_splits=self.n_folds, random_state=None, shuffle=False)
            elif self.fold_type=="TimeSeriesSplit":
                self.splitter = TimeSeriesSplit(n_splits=self.n_folds, max_train_size=None, test_size=None, gap=0)
            else:
                raise WrongInputException(input_name="fold_type",
                                          provided_input=self.fold_type,
                                          allowed_inputs=self.FOLD_TYPE_ALLOWED)
        else:
            raise WrongInputException(input_name="n_folds",
                                      provided_input=self.n_folds,
                                      allowed_inputs=self.N_FOLDS_ALLOWED)
            
            
        # Define the estimator to be cross-validated
        if self.n_models>1:

            check_param_grid(self.param_grid)
            
            if self.n_models>self.max_n_models:
                self.estimator_cv=RandomizedSearchCV(estimator=self.estimator,
                                                     param_distributions=self.param_grid,
                                                     cv=self.splitter,
                                                     n_iter=self.max_n_models)
            else:
                self.estimator_cv=GridSearchCV(estimator=self.estimator,
                                               param_grid=self.param_grid,
                                               cv=self.splitter)
        
        else:
            # If param_grid leads to one single model (n_models==1), there's no need to set of cross validation. In this case, just initialize the model and set parameters
            self.estimator_cv = self.estimator
            self.param_grid = {k: self.param_grid.get(k,None)[0] for k in self.param_grid.keys()}
            
            # Set parameters
            self.estimator_cv.set_params(**self.param_grid)
            
    # --------------------
    # Class variables
    # --------------------
    FOLD_TYPE_ALLOWED = ["KFold", "TimeSeriesSplit"]
    N_FOLDS_ALLOWED = [1, 2, "...", "N"]

    # --------------------
    # Private functions
    # --------------------
    def _update_params(self, old_param, new_param, errors="raise"):
        """ Update 'old_param' with 'new_param'
        """
        # Copy old param
        updated_param = old_param.copy()
        
        for k,v in new_param.items():
            if k in old_param:
                updated_param[k] = v
            else:
                if errors=="raise":
                    raise Exception(f"Parameters {k} not recognized as a default parameter for this estimator")
                else:
                    pass
        return updated_param


    # --------------------
    # Public functions
    # --------------------
    def fit(self,Y,W,X):
        
        # Find masks for the different periods pre- and post-treatment
        self.idx_T0 = W == 0
        self.idx_T1 = W == 1
        
        ## Split data into pre-treatment and post-treatment
        # Pre-treatment (Y0,X)
        self.Y_pre, self.X_pre = Y.loc[self.idx_T0], X.loc[self.idx_T0,:]
        
        # Post-treatment (Y1,X)
        self.Y_post, self.X_post = Y.loc[self.idx_T1], X.loc[self.idx_T1,:]
                
        # Estimate f in Y0 = f(X) + eps
        self.estimator_cv.fit(X=self.X_pre,y=self.Y_pre)
                
        # Predict Y0 post-treatment
        self.Y_post_hat = self.estimator_cv.predict(X=self.X_post)
        
        # Get descriptive statistics of Y both pre- and post-treatment
        self.Y_pre_mean = self.Y_pre.mean()
        self.Y_post_mean = self.Y_post.mean()
        self.Y_post_hat_mean = self.Y_post_hat.mean()
        
        # Compute average treatment effect
        self.average_treatment_effet = self.Y_post_mean - self.Y_post_hat_mean
        
        
    def calculate_average_treatment_effect(self, X_post_treatment=None):
        
        if X_post_treatment is None:
            # Recall ate from fit
            average_treatment_effect = self.average_treatment_effet
            
        else:
            # Predict Y0 post-treatment
            Y_post_hat = self.estimator_cv.predict(X=X_post_treatment)
            
            # Average
            Y_post_hat_mean = Y_post_hat.mean()
            
            # Estimated treatment effect as the difference between means of Y1 and Y0-predicted
            average_treatment_effect = self.Y_post_mean - Y_post_hat_mean
                
        return average_treatment_effect
            
    def bootstrap_ate(self, bootstrap_type="circular", n_bootstrap_samples=1000, block_length=5, conf_int=0.95, X_post_treatment=None):
        
        if X_post_treatment is None:
            Y_post_hat = self.Y_post_hat
        else:
            # Predict Y0 post-treatment
            Y_post_hat = self.estimator_cv.predict(X=X_post_treatment)
                        
        # Difference between Y1 and Y0-predicted
        Y_diff = self.Y_post - Y_post_hat
        
        # Initialize Bootstrap
        bootstrap = Bootstrap(bootstrap_type=bootstrap_type,
                              n_bootstrap_samples=n_bootstrap_samples,
                              block_length=block_length)
        
        # Generate bootstrap samples
        Y_diff_bootstrapped = bootstrap.generate_samples(x=Y_diff)
                
        # Compute mean
        Y_diff_bootstrapped_mean = Y_diff_bootstrapped.mean(axis=0)
        
        # Compute other stats from bootstrap (NOT CURRENTLY USED)
        # Y_diff_bootstrapped_std = Y_diff_bootstrapped.std(axis=0)
        # Y_diff_bootstrapped_sem = Y_diff_bootstrapped.sem(axis=0)
        
        # Confidence interval of mean
        alpha = (1-conf_int)/2
        Y_diff_bootstrapped_ci = Y_diff_bootstrapped_mean.quantile(q=[alpha,1-alpha])
        
        results_bootstrapped = {"mean" : Y_diff_bootstrapped_mean.mean(),
                                "ci_lower" : Y_diff_bootstrapped_ci.iloc[0],
                                "ci_upper" : Y_diff_bootstrapped_ci.iloc[1],
                                "mean_distribution" : Y_diff_bootstrapped_mean,
                                "difference_simulated" : Y_diff_bootstrapped,
                                }

        return results_bootstrapped
        
        
        
        
    
        
    
    
    
    
    
    
    
    
    
    
        
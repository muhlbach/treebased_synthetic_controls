#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold, TimeSeriesSplit

# User
from base.base_synthetic_controls import BaseSyntheticControl
from .estimator.constrained_ols import ConstrainedOLS
from .utils.sanity_check import check_param_grid
from .utils.tools import SingleSplit
from .utils.exceptions import WrongInputException
from .utils.bootstrap import Bootstrap

#------------------------------------------------------------------------------
# Ordinary Synthetic Control Group Method
#------------------------------------------------------------------------------
class SyntheticControl(BaseSyntheticControl):
    """
    This class estimates the average treatment effects by constructing a synthetic control group using constrained OLS
    """
    # --------------------
    # Constructor function
    # --------------------
    # HERE!
    def __init__(self,
                 estimator=ConstrainedOLS(),
                 param_grid={'coefs_lower_bound':0,
                             'coefs_lower_bound_constraint':">=",
                             'coefs_sum_bound':1,
                             'coefs_sum_bound_constraint':"<=",},
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
        super().__init__(
            estimator=estimator,
            param_grid=param_grid,
            cv_params=cv_params,
            n_folds=n_folds,
            fold_type=fold_type,
            max_n_models=max_n_models,
            test_size=test_size,
            verbose=verbose,
            )



#------------------------------------------------------------------------------
# Tree-Based Synthetic Control Group Method
#------------------------------------------------------------------------------
class TreeBasedSyntheticControl(BaseSyntheticControl):
    """
    This class estimates the average treatment effects by constructing a synthetic control group using Random Forests
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
        super().__init__(
            estimator=estimator,
            param_grid=param_grid,
            cv_params=cv_params,
            n_folds=n_folds,
            fold_type=fold_type,
            max_n_models=max_n_models,
            test_size=test_size,
            verbose=verbose,
            )

            
#------------------------------------------------------------------------------
# Elastic Net Synthetic Control Group Method
#------------------------------------------------------------------------------
class ElasticNetSyntheticControl(BaseSyntheticControl):
    """
    This class estimates the average treatment effects by constructing a synthetic control group using Elastic Net
    """
    # --------------------
    # Constructor function
    # --------------------
    def __init__(self,
                 estimator=ElasticNet(fit_intercept=True,
                                      precompute=False,
                                      max_iter=10000,
                                      copy_X=True,
                                      tol=0.0001,
                                      warm_start=False,
                                      positive=False,
                                      random_state=None,
                                      selection='cyclic'),
                 param_grid={"l1_ratio": [1/10, 1/4, 1/2, 3/4, 1],
                             "alpha": np.exp(np.linspace(start=np.log(100), stop=np.log(0.000001), num=100)),
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
        super().__init__(
            estimator=estimator,
            param_grid=param_grid,
            cv_params=cv_params,
            n_folds=n_folds,
            fold_type=fold_type,
            max_n_models=max_n_models,
            test_size=test_size,
            verbose=verbose,
            )
    
    
        
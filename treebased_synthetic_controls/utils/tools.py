#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
import numpy as np
import pickle
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples

#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------
# Save and load objects using pickle
def save_object_by_pickle(path,obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_object_by_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def break_links(x):
    """ Break link to x by copying it"""
    x = x.copy()
    return x

def update_parameters(self, default_params, user_params, element_as_list=True):
    """ Merge default and supplied parameters """
    if bool(user_params):
        for k,v in user_params.items():
            
            # If key is in param_default, then update value
            if k in default_params:
                default_params[k] = v
            else:
                raise Exception(f"Parameter '{k}' is invalid because it is not part of default parameters")
            
    if element_as_list:
        # Turn every element into a list    
        default_params = {key:(value if isinstance(value, list) else [value]) for key,value in default_params.items()}
            
    return default_params    


class SingleSplit(object):
    # --------------------
    # Constructor function
    # --------------------
    def __init__(self,
                 test_size=0.25):
        self.test_size=test_size

    # --------------------
    # CLASS VARIABLES
    # --------------------        
    N_SPLITS = 1
        
    # --------------------
    # Public
    # --------------------        
    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : array-like of shape (n_samples,), default=None
            The target variable for supervised learning problems.
        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)    
        indices = np.arange(n_samples)

        n_test = int(self.test_size*n_samples)
        n_train = n_samples-n_test
        
        train = indices[0:n_train]
        test  = indices[n_train:n_samples]
        
        yield train, test

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.N_SPLITS

#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
import numpy as np
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples

#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------
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



#------------------------------------------------------------------------------
# LEGACY CODE
#------------------------------------------------------------------------------
# class _BaseSingleFold(BaseCrossValidator, metaclass=ABCMeta):
#     """Base class for KFold, GroupKFold, and StratifiedKFold"""

#     @abstractmethod
#     def __init__(self, n_splits, *, shuffle, random_state):
#         if not isinstance(n_splits, numbers.Integral):
#             raise ValueError(
#                 "The number of folds must be of Integral type. "
#                 "%s of type %s was passed." % (n_splits, type(n_splits))
#             )
#         n_splits = int(n_splits)

#         if n_splits <= 1:
#             raise ValueError(
#                 "k-fold cross-validation requires at least one"
#                 " train/test split by setting n_splits=2 or more,"
#                 " got n_splits={0}.".format(n_splits)
#             )

#         if not isinstance(shuffle, bool):
#             raise TypeError("shuffle must be True or False; got {0}".format(shuffle))

#         if not shuffle and random_state is not None:  # None is the default
#             raise ValueError(
#                 "Setting a random_state has no effect since shuffle is "
#                 "False. You should leave "
#                 "random_state to its default (None), or set shuffle=True.",
#             )

#         self.n_splits = n_splits
#         self.shuffle = shuffle
#         self.random_state = random_state

#     def split(self, X, y=None, groups=None):
#         """Generate indices to split data into training and test set.
#         Parameters
#         ----------
#         X : array-like of shape (n_samples, n_features)
#             Training data, where `n_samples` is the number of samples
#             and `n_features` is the number of features.
#         y : array-like of shape (n_samples,), default=None
#             The target variable for supervised learning problems.
#         groups : array-like of shape (n_samples,), default=None
#             Group labels for the samples used while splitting the dataset into
#             train/test set.
#         Yields
#         ------
#         train : ndarray
#             The training set indices for that split.
#         test : ndarray
#             The testing set indices for that split.
#         """
#         X, y, groups = indexable(X, y, groups)
#         n_samples = _num_samples(X)
#         if self.n_splits > n_samples:
#             raise ValueError(
#                 (
#                     "Cannot have number of splits n_splits={0} greater"
#                     " than the number of samples: n_samples={1}."
#                 ).format(self.n_splits, n_samples)
#             )

#         # We only use the LAST fold in the KFold split
#         for fold_id, (train, test) in enumerate(super().split(X, y, groups)):
#             if fold_id+1==self.n_splits:
#                 yield train, test

#     def get_n_splits(self, X=None, y=None, groups=None):
#         return 1


# class SingleFold(_BaseSingleFold):
#     def __init__(self, *, shuffle=False, random_state=None):
#         super().__init__(n_splits=2, shuffle=shuffle, random_state=random_state)

#     def _iter_test_indices(self, X, y=None, groups=None):
#         n_samples = _num_samples(X)
#         indices = np.arange(n_samples)
#         if self.shuffle:
#             check_random_state(self.random_state).shuffle(indices)

#         n_splits = self.n_splits
#         fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
#         fold_sizes[: n_samples % n_splits] += 1
#         current = 0
#         for fold_size in fold_sizes:
#             start, stop = current, current + fold_size
#             yield indices[start:stop]
#             current = stop
           
            
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# User
from ..utils.exceptions import WrongInputException

# Read about score function here:
# https://kiwidamien.github.io/custom-loss-vs-custom-scoring.html
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html

#------------------------------------------------------------------------------
# Constrained OLS
#------------------------------------------------------------------------------
class ConstrainedOLS(BaseEstimator, RegressorMixin):
    """
    Read about solvers: https://www.cvxpy.org/tutorial/advanced/index.html?highlight=scs#choosing-a-solver
    """
    # --------------------
    # Constructor function
    # --------------------
    def __init__(self,
                 coefs_lower_bound = None,
                 coefs_lower_bound_constraint = ">=",
                 coefs_upper_bound = None,
                 coefs_upper_bound_constraint = "<=",
                 coefs_sum_bound = None,
                 coefs_sum_bound_constraint = "<=",
                 verbose = False,
                 ):
        self.coefs_lower_bound = coefs_lower_bound
        self.coefs_lower_bound_constraint = coefs_lower_bound_constraint
        self.coefs_upper_bound = coefs_upper_bound
        self.coefs_upper_bound_constraint = coefs_upper_bound_constraint
        self.coefs_sum_bound = coefs_sum_bound
        self.coefs_sum_bound_constraint = coefs_sum_bound_constraint
        self.verbose = verbose
        
        # Initiate other variables
        self.y_name = ""
        
    # --------------------
    # Class variables
    # --------------------
    LOWER_CONSTRAINTS_ALLOWED = [">=", ">"]
    UPPER_CONSTRAINTS_ALLOWED = ["<=", "<"]
    SUM_CONSTRAINTS_ALLOWED = ["==", ">=", ">","<=", "<"]

    # --------------------
    # Private
    # --------------------
    def _score(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred)

    # --------------------
    # Public functions
    # --------------------
    def fit(self, X, y):
        
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        if isinstance(y, pd.Series):
            self.y_name = y.name
        
        # Break links and transform
        X = np.array(X.copy())
        y = np.array(y.copy())
        
        # Reshape
        y = y.reshape(-1,)
        
        if X.shape[0]!=y.shape[0]:
            raise Exception(f"y is {y.shape}-dim vector, but X is {X.shape}-dim matrix")
        
        # Set up decision variables, i.e. beta coefficients
        beta = cp.Variable(shape=(X.shape[1],), name="beta", integer=False)

        # Initialize constraint    
        constraints = []

        # Lower constraints        
        if self.coefs_lower_bound is not None:
            if self.coefs_lower_bound_constraint == ">=":
                constraints += [beta >= self.coefs_lower_bound]
            elif self.coefs_lower_bound_constraint == ">":
                constraints += [beta > self.coefs_lower_bound]
            else:
                raise WrongInputException(input_name="coefs_lower_bound_constraint",
                                          provided_input=self.coefs_lower_bound_constraint,
                                          allowed_inputs=self.LOWER_CONSTRAINTS_ALLOWED)
                
        # Upper constraints        
        if self.coefs_upper_bound is not None:
            if self.coefs_upper_bound_constraint == "<=":
                constraints += [beta <= self.coefs_upper_bound]
            elif self.coefs_upper_bound_constraint == "<":
                constraints += [beta < self.coefs_upper_bound]
            else:
                raise WrongInputException(input_name="coefs_upper_bound_constraint",
                                          provided_input=self.coefs_upper_bound_constraint,
                                          allowed_inputs=self.UPPER_CONSTRAINTS_ALLOWED)

        # Sum constraints        
        if self.coefs_sum_bound is not None:
            if self.coefs_sum_bound_constraint == "==":
                constraints += [cp.sum(beta) == self.coefs_sum_bound]
            elif self.coefs_sum_bound_constraint == ">=":
                constraints += [cp.sum(beta) >= self.coefs_sum_bound] 
            elif self.coefs_sum_bound_constraint == ">":
                constraints += [cp.sum(beta) > self.coefs_sum_bound]
            elif self.coefs_sum_bound_constraint == "<=":
                constraints += [cp.sum(beta) <= self.coefs_sum_bound]
            elif self.coefs_sum_bound_constraint == "<":
                constraints += [cp.sum(beta) < self.coefs_sum_bound]
            else:
                raise WrongInputException(input_name="coefs_sum_bound_constraint",
                                          provided_input=self.coefs_sum_bound_constraint,
                                          allowed_inputs=self.SUM_CONSTRAINTS_ALLOWED)
            
        # Set up ojective function
        objective = cp.Minimize(cp.sum_squares(y - X @ beta))
        
        # Instantiate
        problem = cp.Problem(objective=objective, constraints=constraints)
        
        # Solve (No need to specify solver because by default CVXPY calls the solver most specialized to the problem type)
        problem.solve(verbose=self.verbose)
        
        # Beta hat
        self.beta_ = beta.value
        
        # Fitted valued
        self.y_fitted_ = X @ self.beta_

        # Return mean squared error
        self.best_score_ = self._score(y_true=y, y_pred=self.y_fitted_)
        
        return self
        
        
    def predict(self, X):
        
        # Check is fit had been called
        check_is_fitted(self)
        
        # Input validation
        X = check_array(X)
        
        # Break links and transform
        X = np.array(X.copy())
        
        y_hat = pd.Series(X @ self.beta_, name=self.y_name)

        return y_hat

    def score(self, X, y, sample_weight=None):
        # Predict
        y_pred = self.predict(X)
        
        return self._score(y_true=y, y_pred=y_pred)
        
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
import numpy as np
import pandas as pd
import cvxpy as cp
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tools.tools import add_constant
from scipy.stats import norm

# User
from .exceptions import WrongInputException

###############################################################################
# Main
###############################################################################

#------------------------------------------------------------------------------
# Tools
#------------------------------------------------------------------------------
def get_colnames(x,prefix="X"):
    try:
        dim = x.shape[1]
        colnames = [prefix+str(j) for j in np.arange(start=1,stop=dim+1)]
    except IndexError:
        colnames = [prefix]
        
    return colnames

def convert_to_dict_series(Yobs=None,Ytrue=None,Y0=None,Y1=None,W=None):
    # Save local arguments
    args = locals()
    
    # Convert values to series with appropriate names
    args = {k: pd.Series(v, name=k) for k,v in args.items() if v is not None}
    
    return args

def convert_to_dict_df(X=None):
    # Save local arguments
    args = locals()
    
    # Convert values to series with appropriate names
    args = {k: pd.DataFrame(v, columns=get_colnames(x=v,prefix=k)) for k,v in args.items() if v is not None}
    
    return args

def convert_normal_to_uniform(x, mu="infer", sigma="infer", lower_bound=0, upper_bound=1, n_digits_round=2):
    """ See link: https://math.stackexchange.com/questions/2343952/how-to-transform-gaussiannormal-distribution-to-uniform-distribution
    """
    # Convert to np and break link
    x = np.array(x.copy())   
       
    if mu=="infer":
        mu = np.mean(x, axis=0).round(n_digits_round)
    if sigma=="infer":
        sigma = np.sqrt(np.var(x, axis=0)).round(n_digits_round)
    
    # Get CDF
    x_cdf = norm.cdf(x=x, loc=mu, scale=sigma)
    
    # Transform
    x_uni = (upper_bound-lower_bound)*x_cdf - lower_bound
    
    return x_uni
    


#------------------------------------------------------------------------------
# Generate X data
#------------------------------------------------------------------------------
def generate_ar_process(T=100, x_p=5, ar_p=3, burnin=50, **kwargs):

    # Extract/generate initial coeffients of X
    mu = kwargs.get('mu', 0)
    sigma = kwargs.get('sigma', 1)

    ## Extract/generate parameters for AR
    const = kwargs.get('const', 0)
    ar_coefs = kwargs.get('ar_coefs', np.linspace(start=0.5, stop=0, num=ar_p, endpoint=False))
    error_coef = kwargs.get('error_coef', 1)

    # Fix AR coefs; flip order and reshape to comformable shape
    ar_coefs = np.flip(ar_coefs).reshape(-1,1)

    # Generate errors
    errors = kwargs.get('errors', np.random.multivariate_normal(mean=np.ones(x_p), 
                                                                cov=np.identity(x_p),
                                                                size=T))    

    # Generate errors for burn-in period
    errors_burnin = np.random.multivariate_normal(mean=np.mean(errors,axis=0), 
                                                  cov=np.cov(errors.T),
                                                  size=burnin)

    errors_all = np.concatenate((errors_burnin,errors))

    # Generate initial value(s)
    X = mu + sigma * np.random.randn(ar_p,x_p)

    # Simulate AR(p) with burn-in included
    for b in range(burnin+T):
        X = np.concatenate((X,
                            const + ar_coefs.T @ X[0:ar_p,:] + error_coef * errors_all[b,0:x_p]),
                           axis=0)

    # Return only the last T observations (we have removed the dependency on the initial draws)
    return X[-T:,]

def generate_iid_process(T=100, x_p=5, distribution="normal", **kwargs):

    # Extract for normal
    mu = kwargs.get('mu', 0)
    sigma = kwargs.get('sigma', 1)
    covariance = kwargs.get('covariance', 0)

    # Extract for uniform
    lower_bound = kwargs.get('lower_bound', 0)
    upper_bound = kwargs.get('upper_bound', 1)

    # Construct variance-covariance matrix
    cov_diag = np.diag(np.repeat(a=sigma**2, repeats=x_p))
    cov_off_diag= np.ones(shape=(x_p,x_p)) * covariance
    np.fill_diagonal(a=cov_off_diag, val=0)
    cov_mat = cov_diag + cov_off_diag

    # Generate X
    if distribution=="normal":
        # Draw from normal distribution
        X = np.random.multivariate_normal(mean=np.repeat(a=mu, repeats=x_p), 
                                          cov=cov_mat,
                                          size=T)    
    elif distribution=="uniform":
        # Draw from uniform distribution
        X = np.random.uniform(low=lower_bound,
                              high=upper_bound,
                              size=(T,x_p))
    else:
        raise WrongInputException(input_name="distribution",
                                  provided_input=distribution,
                                  allowed_inputs=["normal", "uniform"])
                   
    return X

def generate_errors(N=1000, p=5, mu=0, sigma=1, cov_X=0.25, cov_y=0.5):

    # Number of dimensions including y
    n_dim = p+1

    ## Construct variance-covariance matrix
    # Construct diagonal with variance = sigma^2
    cov_diag = np.diag(np.repeat(a=sigma**2, repeats=n_dim))
    
    ## Construct off-diagonal with covariances
    # Fill out for X (and y)
    cov_off_diag = np.ones(shape=(n_dim,n_dim)) * cov_X
    
    # Update y entries
    cov_off_diag[p,:] = cov_off_diag[:,p] = cov_y
    
    # Set diagonal to zero
    np.fill_diagonal(a=cov_off_diag, val=0)
    
    # Update final variance-covariance matrix
    cov_mat = cov_diag + cov_off_diag

    # Generate epsilon
    eps = np.random.multivariate_normal(mean=np.repeat(a=mu, repeats=n_dim), 
                                        cov=cov_mat,
                                        size=N)    

    return eps

#------------------------------------------------------------------------------
# Generate f_star = E[Y|X=x]
#------------------------------------------------------------------------------
def _solve_meta_problem(A,B,w):
    """
    Solve diag(X @ A') = B @ w for X such that X_ij>=0 and sum_j(X_ij)==1 for all i
    """
    # Vectorize weights
    w = _vectorize_beta(beta=w,x=B)
    
    # Set up variable to solve for
    X = cp.Variable(shape=(A.shape))
        
    # Set up constraints
    constraints = [X >= 0,
                   X @ np.ones(shape=(A.shape[1],)) == 1
                   ]
    
    # Set up objective function
    objective = cp.Minimize(cp.sum_squares(cp.diag(X @ A.T) - B @ w))
    
    # Instantiate
    problem = cp.Problem(objective=objective, constraints=constraints)
    
    # Solve (No need to specify solver because by default CVXPY calls the solver most specialized to the problem type)
    problem.solve(verbose=False)
    
    return X.value
    
def _vectorize_beta(beta,x):
    """
    Turn supplied beta into an appropriate shape
    """
    if isinstance(beta, (int, float, np.integer)):
        beta = np.repeat(a=beta, repeats=x.shape[1])        
    elif isinstance(beta, np.ndarray):
        if len(beta)<x.shape[1]:
            beta = np.tile(A=beta, reps=int(np.ceil(x.shape[1]/len(beta))))
        # Shorten potentially
        beta = beta[:x.shape[1]]
    elif isinstance(beta, str):
        if beta=="uniform":
            beta = np.repeat(a=1/x.shape[1], repeats=x.shape[1])
        elif beta=="flip_uniform":
            beta = np.repeat(a=1/x.shape[1], repeats=x.shape[1])            
    else:
        raise WrongInputException(input_name="beta",
                                  provided_input=beta,
                                  allowed_inputs=[int, float, str, np.ndarray, np.integer])       
        
    # Make sure beta has the right dimensions
    beta = beta.reshape(-1,)        
    
    if x.shape[1]!=beta.shape[0]:
            raise Exception(f"Beta is {beta.shape}-dim vector, but X is {x.shape}-dim matrix")
    
    return beta


def generate_linear_data(x,
                         beta=1,
                         beta_handling="default",
                         include_intercept=False,
                         expand=False,
                         degree=2,
                         interaction_only=False,
                         enforce_limits=False,
                         tol_fstar=100,
                         **kwargs):

    #
    BETA_HANDLING_ALLOWED = ["default", "structural", "split_order"]
    
    # Convert to np and break link
    x = np.array(x.copy())    

    # Convert extrama points of X
    if enforce_limits:
        x_min, x_max  = np.min(x, axis=1), np.max(x, axis=1)

    # Series expansion of X
    if expand:        
        if degree<2:
            raise Exception(f"When polynomial features are generated (expand=True), 'degree' must be >=2. It is curently {degree}")
        
        # Instantiate 
        polynomialfeatures = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False, order='C')
    
        # Expand x
        x_poly = polynomialfeatures.fit_transform(x)[:,x.shape[1]:]
        
        # Concatenate
        x_all = np.concatenate((x,x_poly), axis=1)
        
    else:
        x_all = x
        
    # Include a constant in X
    if include_intercept:
        x = add_constant(data=x, prepend=True, has_constant='skip')
        
    # Different ways to generating beta and fstar
    if beta_handling=="default":
        # Make beta a conformable vector
        beta = _vectorize_beta(beta=beta,x=x_all)
                
        # Generate fstar=E[y|X=x]
        f_star = x_all @ beta

    elif beta_handling=="structural":
        # Get tricky weight matrix, solving diag(WX')=X_all*beta_uniform
        weights = _solve_meta_problem(A=x, B=x_all, w="uniform")        

        # Generate fstar=E[y|X=x]
        f_star = np.diagonal(weights @ x.T)
        
        # Fact check this
        f_star_check = x_all @ _vectorize_beta(beta="uniform",x=x_all)

        if np.sum(f_star-f_star_check) > tol_fstar:
            raise Exception("Trickiness didn't work as differences are above tolerance")        
        
    elif beta_handling=="split_order":    

        if isinstance(beta, (int, float, str, np.integer)):
            raise Exception("Whenever 'beta_handling'='split_order', then 'beta' cannot be either (int, float, str)")
        elif len(beta)!=degree:
            raise Exception(f"beta is if length {len(beta)}, but MUST be of length {degree}")
        if not expand:
            raise Exception("Whenever 'beta_handling'='split_order', then 'expand' must be True")
        
        # First-order beta
        beta_first_order = _vectorize_beta(beta=beta[0],x=x)
        
        # Higher-order beta
        beta_higher_order = np.empty(shape=(0,))

        # Initialize
        higher_order_col = 0
        for higher_order in range(2,degree+1):

            # Instantiate 
            poly_temp = PolynomialFeatures(degree=higher_order, interaction_only=interaction_only, include_bias=False, order='C')
    
            # Expand x
            x_poly_temp = poly_temp.fit_transform(x)[:,x.shape[1]+higher_order_col:]

            # Generate temporary betas for this degree of the expansion
            beta_higher_order_temp = _vectorize_beta(beta=beta[higher_order-1],x=x_poly_temp)
                
            # Append betas
            beta_higher_order = np.append(arr=beta_higher_order, values=beta_higher_order_temp)
        
            # Add column counter that governs which columns to match in X
            higher_order_col += x_poly_temp.shape[1]
        
        # Generate fstar=E[y|X=x]
        f_star = x @ beta_first_order + x_poly @ beta_higher_order
        
    else:
        raise WrongInputException(input_name="beta_handling",
                                  provided_input=beta_handling,
                                  allowed_inputs=BETA_HANDLING_ALLOWED)  
        
    # Reshape for conformity
    f_star = f_star.reshape(-1,)
    
    if enforce_limits:
        f_star = np.where(f_star<x_min, x_min, f_star)
        f_star = np.where(f_star>x_max, x_max, f_star)
    
    return f_star
    

def generate_friedman_data_1(x, **kwargs):
    
    # Convert to np and break link
    x = np.array(x.copy())    

    # Sanity check
    if x.shape[1]<5:
        raise Exception(f"Friedman 1 requires at least 5 regresors, but only {x.shape[1]} are provided in x")

    # Generate fstar=E[y|X=x]
    f_star = 0.1*np.exp(4*x[:,0]) + 4/(1+np.exp(-20*(x[:,1]-0.5))) + 3*x[:,2] + 2*x[:,3] + 1*x[:,4]
    
    # Reshape for conformity
    f_star = f_star.reshape(-1,)
    
    return f_star

def generate_friedman_data_2(x, **kwargs):
    
    # Convert to np and break link
    x = np.array(x.copy())    

    # Sanity check
    if x.shape[1]<5:
        raise Exception(f"Friedman 2 requires at least 5 regresors, but only {x.shape[1]} are provided in x")

    # Generate fstar=E[y|X=x]
    f_star = 10*np.sin(np.pi*x[:,0]*x[:,1]) + 20*(x[:,2]-0.5)**2 + 10*x[:,3] + 5*x[:,4]
    
    # Reshape for conformity
    f_star = f_star.reshape(-1,)
    
    return f_star

#------------------------------------------------------------------------------
# Simulate data
#------------------------------------------------------------------------------
def simulate_data(f,
                  T0=500,
                  T1=50,
                  X_type="AR",
                  X_dist="normal",
                  X_dim=5,
                  AR_lags=3,
                  ate=1,
                  eps_mean=0,
                  eps_std=1,
                  eps_cov_x=0,
                  eps_cov_y=0,
                  **kwargs):

    # Total number of time periods
    T = T0 + T1

    # Generate errors
    errors = generate_errors(N=T, p=X_dim, mu=eps_mean, sigma=eps_std, cov_X=eps_cov_x, cov_y=eps_cov_y)
    
    # Generate covariates
    if X_type=="AR":
        X = generate_ar_process(T=T,
                                x_p=X_dim,
                                ar_p=AR_lags,
                                errors=errors)
                
    elif X_type=="iid":
        X = generate_iid_process(T=T,x_p=X_dim,distribution=X_dist, **kwargs)
    
    # Generate W
    W = np.repeat((0,1), (T0,T1))

    # Generate Y
    Y = f(x=X, **kwargs) + ate*W + errors[:,-1]

    # Collect data
    df = pd.concat(objs=[pd.Series(data=Y,name="Y"),
                          pd.Series(data=W,name="W"),
                          pd.DataFrame(data=X,columns=[f"X{d}" for d in range(X.shape[1])])],
                    axis=1)
        
    return df

def generate_data(dgp="AR1", ar_p=1, n_controls=5, T0=500, T1=50, return_as_df=False, **kwargs):
    
    # Valid dgps
    VALID_DGP = ["AR1"]
    
    # Check if dgp is valid
    if not dgp in VALID_DGP:
        raise Exception(f"Choice of input 'dgp' must be one of {VALID_DGP}, but is currently 'dgp'")
    
    # Total number of time periods
    T = T0 + T1
    
    # Generate data
    if dgp=="AR1":
        
        # Number of control units (columns of X)
        x_p=5
        
        # Numbers of lags in AR
        ar_p=2
        
        # Coefficients in AR
        ar_coefs=np.array([0.5, 0.25])
        
        # Errors of X
        errors = np.random.multivariate_normal(mean=np.ones(x_p), 
                                               cov=np.diag(np.ones(x_p)),
                                               size=T)
        
        # Generate X
        X = generate_ar_process(T=T,
                                x_p=x_p,
                                ar_p=ar_p,
                                ar_coefs=ar_coefs,
                                errors=errors)
        
        # beta coefficients as in X*beta
        beta = kwargs.get('beta', np.ones(x_p))


        # tau as in treatment effect Y1-Y0
        tau = kwargs.get('tau', 5)

        # Covariance of eps
        Gamma = kwargs.get('Gamma', np.identity(2))
        
        # Error term
        epsilon = np.random.multivariate_normal(mean=np.zeros(2), cov=Gamma, size=T)

        # Treatment dummy
        W = np.concatenate((np.repeat(0, repeats=T0), np.repeat(1, repeats=T1)), axis=0)
        
        # Potential outcomes
        Y_baseline = X @ beta
        Y0 = Y_baseline + epsilon[:,0]
        Y1 = tau + Y_baseline + epsilon[:,1]
        
        # Observed outcome
        Y_obs = (1-W)*Y0 + W*Y1
    
        # Transform data
        data_output = convert_to_dict_series(Yobs=Y_obs,Ytrue=Y_baseline,Y0=Y0,Y1=Y1,W=W)
        data_input = convert_to_dict_df(X=X)
        
        # Return as df
        df = pd.concat([pd.DataFrame().from_records(data_output),
                        data_input["X"]], axis=1)
        
        # House-keeping
        # del X,Y_obs,Y_baseline,Y0,Y1,W,data_output,data_input
        
        return df
        
    
                                  
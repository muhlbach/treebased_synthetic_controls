#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
import numpy as np
import pandas as pd

# User

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

def generate_iid_process(T=100, x_p=5, **kwargs):

    # Extract/generate initial coeffients of X
    mu = kwargs.get('mu', 0)
    sigma = kwargs.get('sigma', 1)
    covariance = kwargs.get('covariance', 0)

    # Construct variance-covariance matrix
    cov_diag = np.diag(np.repeat(a=sigma**2, repeats=x_p))
    cov_off_diag= np.ones(shape=(x_p,x_p)) * covariance
    np.fill_diagonal(a=cov_off_diag, val=0)
    cov_mat = cov_diag + cov_off_diag

    # Generate X
    X = np.random.multivariate_normal(mean=np.repeat(a=mu, repeats=x_p), 
                                      cov=cov_mat,
                                      size=T)    

    return X


#------------------------------------------------------------------------------
# Generate f_star = E[Y|X=x]
#------------------------------------------------------------------------------


def generate_friedman_data_1(x, **kwargs):
    
    # Convert to np
    x = np.array(x)
    
    # Generate fstar=E[y|X=x]
    f_star = 0.1*np.exp(4*x[:,0]) + 4/(1+np.exp(-20*(x[:,1]-0.5))) + 3*x[:,2] + 2*x[:,3] + 1*x[:,4]
    
    return f_star

def generate_friedman_data_2(x, **kwargs):
    
    # Convert to np
    x = np.array(x)
    
    # Generate fstar=E[y|X=x]
    f_star = 10*np.sin(np.pi*x[:,0]*x[:,1]) + 20*(x[:,2]-0.5)**2 + 10*x[:,3] + 5*x[:,4]
    
    return f_star

#------------------------------------------------------------------------------
# Simulate Y
#------------------------------------------------------------------------------
# HERE !!!
def simulate_data(f, error_distribution, X_type="AR", ate=1, T0=500, T1=50, X_dim=5, AR_lags=3, **kwargs)

    # Total number of time periods
    T = T0 + T1

    # Generate errors
    errors = error_distribution(T=T, p=X_dim, mean=0, variance=1, covariance=0)

    # Generate covariates
    if X_type=="AR":
        X = generate_ar_process(T=T,
                            x_p=X_dim,
                            ar_p=AR_lags,
                            errors=errors)
                
    elif X_type=="iid":
        X = generate_iid_process(T=T,x_p=X_dim)
    
    # Generate W
    W = np.repeat((0,1), (T0,T1))

    # Generate Y
    Y = f(x=X) + ate*W + errors[:,-1]







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
        
    
                                  
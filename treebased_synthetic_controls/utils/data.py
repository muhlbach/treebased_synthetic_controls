#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
import numpy as np
import pandas as pd

# User

#------------------------------------------------------------------------------
# Main
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
                                                                        cov=np.identity(x_p), #Same np.diag(np.ones(x_p))
                                                                        size=T))    

    # Generate initial value(s)
    X = mu + sigma * np.random.randn(ar_p,x_p)

    # Simulate AR(p) with burn-in included
    for b in range(burnin+T):
        X = np.concatenate((X, const + ar_coefs.T @ X[0:ar_p,:] + error_coef * errors[np.random.randint(errors.shape[0], size=1),:]),
                           axis=0)

    # Return only the last T observations (we have removed the dependency on the initial draws)
    return X[-T:,]


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
        
    
                                  
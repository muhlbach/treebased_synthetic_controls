#------------------------------------------------------------------------------
# Run interactively
#------------------------------------------------------------------------------
"""
If running program interactively, please uncomment the lines below.
But first, change the 'path_to_here' variable to your local folder!
"""
# import os
# # Manually set path of current file
# path_to_here = "/Users/muhlbach/Repositories/treebased_synthetic_controls/treebased_synthetic_controls/"
# # Change path
# os.chdir(path_to_here)
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
import pandas as pd
import time

# User
from tbsc import SyntheticControl
from utils import data

#------------------------------------------------------------------------------
# Simple example
#------------------------------------------------------------------------------
# Generate data
df = data.generate_data(dgp="AR1",
                        ar_p=1,
                        n_controls=5,
                        T0=500,
                        T1=500)

Y = df["Yobs"]
W = df["W"]
X = df[[col for col in df.columns if "X" in col]]

# Instantiate SC-object
syntheticcontrol = SyntheticControl(max_n_models=5, n_folds=1)

# Start timer
t0 = time.time()

# Fit
syntheticcontrol.fit(Y=Y,W=W,X=X)
print(f"Estimated ATE: {np.around(syntheticcontrol.average_treatment_effet,2)}")

# Bootstrap
bootstrapped_results = syntheticcontrol.bootstrap_ate()

# Stop timer
t1 = time.time()

#------------------------------------------------------------------------------
# The End
#------------------------------------------------------------------------------
print(f"""**************************************************\nCode finished in {np.round(t1-t0, 1)} seconds\n**************************************************""")

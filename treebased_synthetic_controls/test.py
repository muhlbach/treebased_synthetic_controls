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
                        T1=50)

Y = df["Yobs"]
W = df["W"]
X = df[[col for col in df.columns if "X" in col]]

# Instantiate SC-object
syntheticcontrol = SyntheticControl(max_n_models=5, n_folds=1)

# self = syntheticcontrol

t0 = time.time()
syntheticcontrol.fit(Y=Y,W=W,X=X)
t1 = time.time()

syntheticcontrol.calculate_average_treatment_effect()

print("Code took:", t1-t0)

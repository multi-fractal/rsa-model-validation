"""
generate_sobol_g.py

Generate independent training and test datasets for the Sobol' G-function.

The last column of each output CSV contains the model response Y.
All previous columns correspond to the input variables X1...Xd.
"""

import numpy as np
import pandas as pd

# ==========================================================
# Parameters
# ==========================================================

N_TRAIN = 5000
N_TEST = 5000
N_VARIABLES = 8
RANDOM_STATE = 42

# Classical Sobol' G coefficients
A = np.array([0, 1, 4.5, 9, 99, 99, 99, 99], dtype=float)

# ==========================================================
# Sobol' G-function
# ==========================================================

def sobol_g(X, a):
    """
    Evaluate the Sobol' G-function.

    Parameters
    ----------
    X : ndarray of shape (N, d)
        Input variables in [0, 1].

    a : ndarray of shape (d,)
        Difficulty coefficients.

    Returns
    -------
    ndarray
        Model response.
    """
    term = (np.abs(4.0 * X - 2.0) + a) / (1.0 + a)
    return np.prod(term, axis=1)

# ==========================================================
# Generate datasets
# ==========================================================

rng = np.random.default_rng(RANDOM_STATE)

# Independent training set
X_train = rng.uniform(
    low=0.0,
    high=1.0,
    size=(N_TRAIN, N_VARIABLES)
)

Y_train = sobol_g(X_train, A)

# Independent test set
X_test = rng.uniform(
    low=0.0,
    high=1.0,
    size=(N_TEST, N_VARIABLES)
)

Y_test = sobol_g(X_test, A)

# ==========================================================
# Save datasets
# ==========================================================

columns = [f"X{i+1}" for i in range(N_VARIABLES)] + ["Y"]

df_train = pd.DataFrame(
    np.column_stack((X_train, Y_train)),
    columns=columns
)

df_test = pd.DataFrame(
    np.column_stack((X_test, Y_test)),
    columns=columns
)

df_train.to_csv("sobol_g_train.csv", index=False)
df_test.to_csv("sobol_g_test.csv", index=False)

print("Training dataset saved to sobol_g_train.csv")
print(df_train.head())

print("\nTest dataset saved to sobol_g_test.csv")
print(df_test.head())
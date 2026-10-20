"""
generate_morris.py

Generate independent training and test datasets
for the Morris test function.

Implementation translated from the R package "sensitivity".

The last column of each CSV contains the model response Y.
"""

import numpy as np
import pandas as pd

# ==========================================================
# Parameters
# ==========================================================

N_TRAIN = 5000
N_TEST = 5000
N_VARIABLES = 20
RANDOM_STATE = 42

rng = np.random.default_rng(RANDOM_STATE)

# ==========================================================
# Random coefficients (generated once)
# ==========================================================

b0 = rng.normal()

b1 = np.concatenate([
    np.full(10, 20.0),
    rng.normal(size=10)
])

b2 = rng.normal(size=(20, 20))
b2[:6, :6] = -15.0

b3 = np.zeros((20, 20, 20))
b3[:5, :5, :5] = -10.0

b4 = np.zeros((20, 20, 20, 20))
b4[:4, :4, :4, :4] = 5.0

# ==========================================================
# Morris function
# ==========================================================

def morris_function(X):
    """
    Morris non-monotonic benchmark function.

    Parameters
    ----------
    X : ndarray (N,20)

    Returns
    -------
    ndarray (N,)
    """

    X = np.asarray(X)

    if X.shape[1] != 20:
        raise ValueError("Morris function requires exactly 20 variables.")

    # ---------------------------------------------
    # Variable transformation
    # ---------------------------------------------

    w = 2.0 * (X - 0.5)

    special = [2, 4, 6]      # X3, X5, X7

    w[:, special] = 2.0 * (
        1.1 * X[:, special] /
        (X[:, special] + 0.1)
        - 0.5
    )

    # ---------------------------------------------
    # Constant
    # ---------------------------------------------

    y = np.full(len(X), b0)

    # ---------------------------------------------
    # Linear terms
    # ---------------------------------------------

    for i in range(20):
        y += b1[i] * w[:, i]

    # ---------------------------------------------
    # Two-way interactions
    # ---------------------------------------------

    for i in range(19):

        wi = w[:, i]

        for j in range(i + 1, 20):
            y += b2[i, j] * wi * w[:, j]

    # ---------------------------------------------
    # Three-way interactions
    # ---------------------------------------------

    for i in range(18):

        wi = w[:, i]

        for j in range(i + 1, 19):

            wij = wi * w[:, j]

            for k in range(j + 1, 20):
                y += b3[i, j, k] * wij * w[:, k]

    # ---------------------------------------------
    # Four-way interactions
    # ---------------------------------------------

    for i in range(17):

        wi = w[:, i]

        for j in range(i + 1, 18):

            wij = wi * w[:, j]

            for k in range(j + 1, 19):

                wijk = wij * w[:, k]

                for l in range(k + 1, 20):
                    y += b4[i, j, k, l] * wijk * w[:, l]

    return y

# ==========================================================
# Generate datasets
# ==========================================================

X_train = rng.uniform(
    low=0.0,
    high=1.0,
    size=(N_TRAIN, N_VARIABLES)
)

Y_train = morris_function(X_train)

X_test = rng.uniform(
    low=0.0,
    high=1.0,
    size=(N_TEST, N_VARIABLES)
)

Y_test = morris_function(X_test)

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

df_train.to_csv("morris_train.csv", index=False)
df_test.to_csv("morris_test.csv", index=False)

print("Training dataset saved to morris_train.csv")
print(df_train.head())

print("\nTest dataset saved to morris_test.csv")
print(df_test.head())
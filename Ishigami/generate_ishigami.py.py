"""
generate_ishigami.py

Generate independent training and test datasets
for the Ishigami benchmark function.

Output files:
    ishigami_train.csv
    ishigami_test.csv

Columns:
    x1
    x2
    x3
    y
"""

import numpy as np
import pandas as pd


# ==========================================================
# Ishigami function
# ==========================================================

def ishigami_function(X, a=7, b=0.1):
    """
    Ishigami function.

    Parameters
    ----------
    X : ndarray, shape (n_samples, 3)

    a : float
        Parameter a

    b : float
        Parameter b

    Returns
    -------
    ndarray
        Function values
    """

    x1 = X[:, 0]
    x2 = X[:, 1]
    x3 = X[:, 2]

    y = (
        np.sin(x1)
        + a * np.sin(x2) ** 2
        + b * x3 ** 4 * np.sin(x1)
    )

    return y


# ==========================================================
# Generate datasets
# ==========================================================

def generate_ishigami_data(
        n_samples,
        a=7,
        b=0.1,
        seed=42
):
    """
    Generate Ishigami dataset.

    Parameters
    ----------
    n_samples : int
        Number of samples

    a, b :
        Ishigami parameters

    seed : int
        Random seed

    Returns
    -------
    pandas.DataFrame
    """

    rng = np.random.default_rng(seed)

    # Generate input variables
    X = rng.uniform(
        low=-np.pi,
        high=np.pi,
        size=(n_samples, 3)
    )

    # Calculate true response
    y = ishigami_function(
        X,
        a=a,
        b=b
    )

    df = pd.DataFrame(
        X,
        columns=[
            "x1",
            "x2",
            "x3"
        ]
    )

    df["y"] = y

    return df


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    N_TRAIN = 5000
    N_TEST = 5000

    SEED = 42

    # Independent training dataset
    df_train = generate_ishigami_data(
        n_samples=N_TRAIN,
        seed=SEED
    )

    # Independent test dataset
    df_test = generate_ishigami_data(
        n_samples=N_TEST,
        seed=SEED + 1
    )

    # Save
    df_train.to_csv(
        "ishigami_train.csv",
        index=False
    )

    df_test.to_csv(
        "ishigami_test.csv",
        index=False
    )

    print("Training dataset saved: ishigami_train.csv")
    print(df_train.head())

    print()

    print("Test dataset saved: ishigami_test.csv")
    print(df_test.head())

    print()

    print("Train statistics:")
    print(df_train.describe())

    print()

    print("Test statistics:")
    print(df_test.describe())
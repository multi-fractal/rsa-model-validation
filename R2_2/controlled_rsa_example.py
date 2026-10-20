"""
controlled_rsa_example.py

Controlled experiment:
True model vs Good surrogate vs Bad surrogate.

Goal:
Demonstrate that models with similar prediction accuracy
can have different regional sensitivity structures.
"""

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr, spearmanr

from regional_sensitivity import rsa_pairwise_distances


# ==========================================================
# Parameters
# ==========================================================

N = 10000
SEED = 42

rng = np.random.default_rng(SEED)


# ==========================================================
# Generate input variables
# ==========================================================

X1 = rng.uniform(-2, 2, N)
X2 = rng.uniform(-2, 2, N)
X3 = rng.uniform(-2, 2, N)

X = np.column_stack(
    [
        X1,
        X2,
        X3
    ]
)


# ==========================================================
# True model
# ==========================================================

noise = rng.normal(
    0,
    0.1,
    N
)

Y_true = (
    X1**3
    + 0.1*X2
    + noise
)


# ==========================================================
# Good surrogate
# preserves the original structure
# ==========================================================

Y_good = (
    0.90*X1**3
    + 0.11*X2
)


# ==========================================================
# Bad surrogate
# similar accuracy but wrong regional structure
# ==========================================================

Y_bad = (
    0.95*X1**3
    + 0.1*X2
    + 0.25*X3
)


# ==========================================================
# Save datasets
# ==========================================================

def save_dataset(filename, Y):

    df = pd.DataFrame(
        {
            "X1": X1,
            "X2": X2,
            "X3": X3,
            "Y": Y
        }
    )

    df.to_csv(
        filename,
        index=False
    )


save_dataset(
    "true_model.csv",
    Y_true
)

save_dataset(
    "good_model.csv",
    Y_good
)

save_dataset(
    "bad_model.csv",
    Y_bad
)


# ==========================================================
# Prediction metrics
# ==========================================================

def prediction_metrics(Y):

    r2 = r2_score(
        Y_true,
        Y
    )

    rmse = np.sqrt(
        mean_squared_error(
            Y_true,
            Y
        )
    )

    return r2, rmse


# ==========================================================
# RSA calculation
# ==========================================================

true_rsa, _, _ = rsa_pairwise_distances(
    X,
    Y_true,
    ngroup=10
)


def rsa_metrics(Y):

    rsa, _, _ = rsa_pairwise_distances(
        X,
        Y,
        ngroup=10
    )

    a = true_rsa.flatten()
    b = rsa.flatten()

    return {
        "RSA_RMSE":
            np.sqrt(
                np.mean(
                    (a-b)**2
                )
            ),

        "RSA_Pearson":
            pearsonr(a, b)[0],

        "RSA_Spearman":
            spearmanr(a, b)[0],

        "RSA_Cosine":
            np.dot(a,b) /
            (
                np.linalg.norm(a)
                *
                np.linalg.norm(b)
                +
                1e-12
            )
    }


# ==========================================================
# Comparison
# ==========================================================

results = []


models = {
    "Ideal model": X1**3 + 0.1*X2,
    "Good surrogate": Y_good,
    "Bad surrogate": Y_bad
}


for name, Y in models.items():

    r2, rmse = prediction_metrics(Y)

    rsa = rsa_metrics(Y)

    results.append(
        {
            "Model": name,
            "R2": r2,
            "RMSE": rmse,
            **rsa
        }
    )


results_df = pd.DataFrame(results)


print("\nResults")
print("="*90)

print(
    results_df.to_string(
        index=False
    )
)


results_df.to_csv(
    "controlled_rsa_results.csv",
    index=False
)


print("\nSaved:")
print("true_model.csv")
print("good_model.csv")
print("bad_model.csv")
print("controlled_rsa_results.csv")
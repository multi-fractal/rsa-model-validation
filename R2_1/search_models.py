"""
search_models.py

Find surrogate models with similar prediction accuracy
but different regional sensitivity structure.

Pipeline:
1. Generate true model
2. Generate random surrogate models
3. Evaluate R2 and RMSE
4. Compute RSA similarity
5. Select models with similar R2 but different RSA
"""

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, mean_squared_error

from regional_sensitivity import rsa_pairwise_distances
from scipy.stats import pearsonr, spearmanr


# ==========================================================
# Parameters
# ==========================================================

N = 10000
N_MODELS = 5000

RANDOM_STATE = 42

TARGET_R2 = 0.95
R2_TOL = 0.005


# ==========================================================
# Generate data
# ==========================================================

rng = np.random.default_rng(RANDOM_STATE)


X1 = rng.uniform(-2, 2, N)
X2 = rng.uniform(-2, 2, N)
X3 = rng.uniform(-2, 2, N)


noise = rng.normal(
    0,
    0.1,
    N
)


X = np.column_stack(
    (X1, X2, X3)
)


# True model

Y_true = (
    X1**3
    + 0.1*X2
    + noise
)


df_true = pd.DataFrame(
    {
        "X1": X1,
        "X2": X2,
        "X3": X3,
        "Y": Y_true
    }
)


df_true.to_csv(
    "true_model.csv",
    index=False
)


# ==========================================================
# True RSA
# ==========================================================

true_rsa, _, _ = rsa_pairwise_distances(
    X,
    Y_true,
    ngroup=10
)


# ==========================================================
# Generate candidate models
# ==========================================================

results = []

models = []


for i in range(N_MODELS):

    a = rng.uniform(0.5, 1.5)
    b = rng.uniform(-0.3, 0.3)
    c = rng.uniform(-0.2, 0.2)
    d = rng.uniform(-0.2, 0.2)
    e = rng.uniform(-1, 1)


    Y_model = (
        a*X1**3
        + b*X2
        + c*X2**2
        + d*np.sin(X2)
        + e
    )


    r2 = r2_score(
        Y_true,
        Y_model
    )

    rmse = np.sqrt(
        mean_squared_error(
            Y_true,
            Y_model
        )
    )


    # keep only interesting models
    if abs(r2 - TARGET_R2) < R2_TOL:

        rsa_model, _, _ = rsa_pairwise_distances(
            X,
            Y_model,
            ngroup=10
        )


        a_vec = true_rsa.flatten()
        b_vec = rsa_model.flatten()


        rsa_rmse = np.sqrt(
            np.mean(
                (a_vec-b_vec)**2
            )
        )

# RSA similarity metrics

        pearson = pearsonr(
            a_vec,
            b_vec
            )[0]

        spearman = spearmanr(
            a_vec,
            b_vec
            )[0]

        cosine = np.dot(
            a_vec,
            b_vec
            ) / (
                np.linalg.norm(a_vec)
                *
                np.linalg.norm(b_vec)
                +
                1e-12
        )

        results.append(
            {
                "id": i,
                "a": a,
                "b": b,
                "c": c,
                "d": d,
                "e": e,
                "R2": r2,
                "RMSE": rmse,
                "RSA_RMSE": rsa_rmse,
                "RSA_Pearson": pearson,
                "RSA_Spearman": spearman,
                "RSA_Cosine": cosine                
            }
        )


        models.append(
            {
                "id": i,
                "Y": Y_model
            }
        )


print(
    f"Found models: {len(results)}"
)


# ==========================================================
# Save candidates
# ==========================================================

results_df = pd.DataFrame(
    results
)


results_df.to_csv(
    "candidate_models.csv",
    index=False
)


if len(results_df) < 2:

    raise RuntimeError(
        "Not enough models found. Increase tolerance."
    )


# ==========================================================
# Select models
# ==========================================================

best_rsa = results_df.sort_values(
    "RSA_RMSE"
).iloc[0]


worst_rsa = results_df.sort_values(
    "RSA_RMSE"
).iloc[-1]


print("\nSelected models")
print("================")

print("\nBest RSA preservation:")
print(best_rsa)


print("\nWorst RSA preservation:")
print(worst_rsa)


# ==========================================================
# Save selected datasets
# ==========================================================

for name, model_info in [
    ("best_rsa_model.csv", best_rsa),
    ("worst_rsa_model.csv", worst_rsa)
]:

    Y = None

    idx = int(
        model_info["id"]
    )

    # regenerate model

    Y = (
        model_info["a"]*X1**3
        + model_info["b"]*X2
        + model_info["c"]*X2**2
        + model_info["d"]*np.sin(X2)
        + model_info["e"]
    )


    df = pd.DataFrame(
        {
            "X1": X1,
            "X2": X2,
            "X3": X3,
            "Y": Y
        }
    )


    df.to_csv(
        name,
        index=False
    )


print("\nDone.")
print("Files created:")
print(" true_model.csv")
print(" candidate_models.csv")
print(" best_rsa_model.csv")
print(" worst_rsa_model.csv")
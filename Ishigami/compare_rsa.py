"""
compare_rsa_ishigami.py

Comparison of RSA (pairwise KS-distance matrices)
between the true Ishigami function and surrogate models.

Separate comparisons are performed for the training
and test datasets.
"""

import numpy as np
import pandas as pd

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

from regional_sensitivity import (
    load_data_from_csv,
    rsa_pairwise_distances
)


# ==========================================================
# RSA comparison
# ==========================================================

def compare_models(ks_A, ks_B):

    if ks_A.shape != ks_B.shape:
        raise ValueError(
            f"RSA matrices have different shapes: "
            f"{ks_A.shape} vs {ks_B.shape}"
        )

    a = ks_A.flatten()
    b = ks_B.flatten()

    pearson = pearsonr(a, b)[0]
    spearman = spearmanr(a, b)[0]

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    cosine = np.dot(a, b) / (
        norm_a * norm_b + 1e-10
    )

    rmse = np.sqrt(
        mean_squared_error(a, b)
    )

    mae = mean_absolute_error(
        a,
        b
    )

    return {
        "Pearson": pearson,
        "Spearman": spearman,
        "Cosine": cosine,
        "RMSE": rmse,
        "MAE": mae
    }


# ==========================================================
# One dataset (train or test)
# ==========================================================

def compare_dataset(dataset):

    print("=" * 70)
    print(f"Processing {dataset.upper()} dataset")
    print("=" * 70)

    files = {

        "True Ishigami":
            f"ishigami_{dataset}.csv",

        "Linear":
            f"ishigami_linear_{dataset}_generated.csv",

        "Polynomial":
            f"ishigami_polynomial_{dataset}_generated.csv",

        "Random Forest":
            f"ishigami_random_forest_{dataset}_generated.csv",

        "XGBoost":
            f"ishigami_xgboost_{dataset}_generated.csv",
    }


    rsa_results = {}


    # ------------------------------------------------------
    # Calculate RSA matrices
    # ------------------------------------------------------

    for name, file in files.items():

        print(f"Calculating RSA: {name}")

        X, Y, _ = load_data_from_csv(file)

        ks_matrix, _, _ = rsa_pairwise_distances(
            X,
            Y,
            ngroup=10
        )

        rsa_results[name] = ks_matrix


    true_rsa = rsa_results["True Ishigami"]


    comparison = []


    # ------------------------------------------------------
    # Compare surrogate models
    # ------------------------------------------------------

    for name, model_rsa in rsa_results.items():

        if name == "True Ishigami":
            continue


        result = compare_models(
            true_rsa,
            model_rsa
        )

        result["Model"] = name

        comparison.append(result)


    comparison_df = pd.DataFrame(
        comparison
    )


    comparison_df = comparison_df[
        [
            "Model",
            "Spearman",
            "Pearson",
            "Cosine",
            "RMSE",
            "MAE",
        ]
    ]


    print()

    print(
        comparison_df.to_string(
            index=False
        )
    )


    out_file = (
        f"rsa_comparison_ishigami_{dataset}.csv"
    )

    comparison_df.to_csv(
        out_file,
        index=False
    )


    print(
        f"\nSaved: {out_file}"
    )


# ==========================================================
# Main
# ==========================================================

def main():

    compare_dataset("train")

    print()

    compare_dataset("test")


if __name__ == "__main__":
    main()
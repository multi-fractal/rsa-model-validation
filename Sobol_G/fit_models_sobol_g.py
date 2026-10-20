"""
fit_models_sobol_g.py

Train surrogate models for the Sobol' G-function and generate
prediction datasets for RSA comparison on both the training
and test datasets.
"""

import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from xgboost import XGBRegressor

# ==========================================================
# Load datasets
# ==========================================================

df_train = pd.read_csv("sobol_g_train.csv")
df_test = pd.read_csv("sobol_g_test.csv")

X_train = df_train.iloc[:, :-1].values
Y_train = df_train.iloc[:, -1].values

X_test = df_test.iloc[:, :-1].values
Y_test = df_test.iloc[:, -1].values

# ==========================================================
# Models
# ==========================================================

models = {

    "linear": LinearRegression(),

    "polynomial": Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("linear", LinearRegression())
    ]),

    "random_forest": RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1
    ),

    "xgboost": XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42
    )
}

# ==========================================================
# Train models
# ==========================================================

for name, model in models.items():

    print("=" * 60)
    print(f"Training {name}")

    # ------------------------------------------------------
    # Train
    # ------------------------------------------------------

    model.fit(X_train, Y_train)

    # ------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    r2_train = r2_score(Y_train, pred_train)
    r2_test = r2_score(Y_test, pred_test)

    print(f"Train R² = {r2_train:.6f}")
    print(f" Test R² = {r2_test:.6f}")

    # ------------------------------------------------------
    # Save train predictions
    # ------------------------------------------------------

    out_train = df_train.copy()
    out_train.iloc[:, -1] = pred_train

    filename_train = f"sobol_{name}_train_generated.csv"
    out_train.to_csv(filename_train, index=False)

    # ------------------------------------------------------
    # Save test predictions
    # ------------------------------------------------------

    out_test = df_test.copy()
    out_test.iloc[:, -1] = pred_test

    filename_test = f"sobol_{name}_test_generated.csv"
    out_test.to_csv(filename_test, index=False)

    print(f"Saved: {filename_train}")
    print(f"Saved: {filename_test}")
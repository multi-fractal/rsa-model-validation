# Machine Learning Model Validation Through Regional Sensitivity Analysis (RSA)

This repository contains the Python implementation for validating machine learning surrogate models using Regional Sensitivity Analysis (RSA) based on complete pairwise Kolmogorov-Smirnov (KS) distance matrices. 

The scripts provided focus on the **Ishigami function** benchmark.

---

## Repository Structure

1. **`generate_ishigami.py`**  
   Generates independent training and test datasets using the analytical Ishigami function and saves them as CSV files.

2. **`fit_models_ishigami.py`**  
   Trains four predictive models (Linear Regression, Polynomial Regression, Random Forest, and XGBoost) on the training data and generates predictions for both training and test sets.

3. **`regional_sensitivity.py`**  
   Core module implementing the RSA framework: response grouping, empirical CDF calculation, and the computation of complete pairwise KS-distance matrices.

4. **`compare_rsa_ishigami.py`**  
   Computes RSA matrices for the true data and all trained models, then evaluates structural similarity using Pearson correlation, Spearman correlation, Cosine similarity, RMSE, and MAE.

Execution Order
To reproduce the workflow from scratch, run the scripts in the following sequential order:

Step 1: Generate Datasets
Run the data generation script to create the reference training and test files:

Bash
python generate_ishigami.py
Outputs: ishigami_train.csv, ishigami_test.csv

Step 2: Train Models and Generate Predictions
Train the surrogate models and export their predictions:

Bash
python fit_models_ishigami.py
Outputs: ishigami_<model>_train_generated.csv and ishigami_<model>_test_generated.csv for each model.

Step 3: Run RSA Comparison
Compute the pairwise KS-distance matrices and compare the models against the true reference:

Bash
python compare_rsa_ishigami.py
Outputs: Console summary and rsa_comparison_ishigami_train.csv, rsa_comparison_ishigami_test.csv.
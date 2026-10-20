# Controlled RSA Experiment: The Accuracy-Sensitivity Paradox

This directory contains a controlled experiment demonstrating that **machine learning models with similar prediction accuracy can have completely different regional sensitivity structures**. 

It isolates the core concept of the Regional Sensitivity Analysis (RSA) framework: standard predictive metrics ($R^2$, RMSE) can look great while a model has structurally flawed internal dependencies (e.g., picking up a spurious variable).

---

## File Structure

1. **`controlled_rsa_example.py`**  
   Runs the controlled experiment: defines an analytical true model, a "good surrogate" preserving the true structure, and a "bad surrogate" that achieves similar accuracy by introducing a false dependency on an irrelevant variable ($X_3$). Computes both prediction metrics and RSA-based distance metrics.

2. **`regional_sensitivity.py`**  
   Core module providing response grouping, empirical CDF calculation, and complete pairwise Kolmogorov-Smirnov (KS) distance matrices.

---

## Execute the controlled experiment script:

Bash
python controlled_rsa_example.py

Outputs
The script prints a comparison summary to the console and generates the following files:

true_model.csv, good_model.csv, bad_model.csv: Generated synthetic datasets for the analytical experiment.

controlled_rsa_results.csv: Summary table comparing prediction accuracy (R2, RMSE) against RSA structural distance metrics (RMSE, Pearson, Spearman, and Cosine similarity).
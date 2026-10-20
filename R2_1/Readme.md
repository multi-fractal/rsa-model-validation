# Surrogate Model Search Pipeline: Exploring the Accuracy-Sensitivity Paradox

This directory contains a Monte Carlo-based search pipeline to explore the relationship between predictive accuracy and internal structural consistency using Regional Sensitivity Analysis (RSA). 

It demonstrates that within a pool of surrogate models sharing an identical high level of predictive accuracy (e.g., $R^2 \approx 0.95$), one can find both models that preserve the true internal sensitivity structure and models that severely distort it.


## File Structure

1. **`search_models.py`**  
   The main pipeline script. It generates a true model, runs a Monte Carlo search across thousands of random surrogate candidates, filters them by target $R^2$ tolerance, evaluates their RSA distance metrics against the true model, and isolates the best and worst structural preservation cases.
2. **`regional_sensitivity.py`**  
   Core math module providing response grouping, empirical CDF calculation, and pairwise Kolmogorov-Smirnov (KS) distance matrices.

---

How to RunExecute the search pipeline:Bashpython search_models.py

OutputsThe script prints summary diagnostics to the console and generates the following files in the working directory:true_model.csv: The baseline synthetic dataset generated from the analytical true model.candidate_models.csv: A comprehensive table of all evaluated surrogate models passing the $R^2$ threshold, including their parameters, prediction errors, and RSA distance metrics.best_rsa_model.csv: Dataset for the surrogate model with the best structural preservation (minimum RSA-RMSE).worst_rsa_model.csv: Dataset for the surrogate model with the worst structural preservation (maximum RSA-RMSE) despite having equivalent predictive accuracy.
# regional_sensitivity.py

import numpy as np
import pandas as pd
from warnings import warn

# ==========================================
# 1. Загрузка данных
# ==========================================

def load_data_from_csv(filepath):

    df = pd.read_csv(filepath)
    
    data = df.values
    X = data[:, :-1]  
    Y = data[:, -1]   
    
    feature_names = df.columns[:-1].tolist()
    
    return X, Y, feature_names

# ==========================================
# 2. Математика RSA (полностью автономная)
# ==========================================

def empiricalcdf(x, xx):
    x = np.asarray(x)
    n = len(x)
    if n == 0:
        return np.zeros_like(xx, dtype=float)
    x_sorted = np.sort(x)
    counts = np.searchsorted(x_sorted, xx, side='right')
    return counts / n

def split_sample(Y, ngroup=10):
    """
    Split the response variable into approximately equiprobable groups.

    Parameters
    ----------
    Y : ndarray
        Response variable.

    ngroup : int
        Requested number of groups.

    Returns
    -------
    idx : ndarray
        Group index (0 ... ngroup_eff-1) for each observation.

    bounds : ndarray
        Group boundaries.

    centers : ndarray
        Mean response value within each group.

    ngroup_eff : int
        Effective number of groups.
    """

    Y = np.asarray(Y).flatten()
    N = len(Y)

    n_eff = ngroup

    # ---------------------------------------------------------
    # Discrete response
    # ---------------------------------------------------------

    unique_values = np.unique(Y)

    if len(unique_values) <= ngroup:

        n_eff = len(unique_values)

        centers = np.sort(unique_values)

        bounds = np.concatenate((centers, [centers[-1]]))

    # ---------------------------------------------------------
    # Continuous response
    # ---------------------------------------------------------

    else:

        Y_sorted = np.sort(Y)

        split = [int(round(i)) for i in np.linspace(0, N, n_eff + 1)]
        split[-1] = N - 1

        bounds = Y_sorted[split]

        keep = np.ones(len(bounds), dtype=bool)

        for k in range(len(bounds)):

            if not keep[k]:
                continue

            repeated = np.sum(bounds[keep] == bounds[k])

            if repeated > 2:

                keep[k] = False

            elif repeated == 2:

                if k < len(bounds) - 3:

                    keep[k + 2] = False

                    next_value = Y_sorted[np.where(Y_sorted > bounds[k])[0][0]]

                    bounds[k + 1] = next_value

                elif k == len(bounds) - 3:

                    next_value = Y_sorted[np.where(Y_sorted > bounds[k])[0][0]]

                    bounds[k + 1] = next_value

        bounds = bounds[keep]

        n_eff = len(bounds) - 1

    # ---------------------------------------------------------
    # Assign observations to groups
    # ---------------------------------------------------------

    idx = np.full(N, -1, dtype=int)

    for k in range(n_eff):

        if k < n_eff - 1:

            mask = (Y >= bounds[k]) & (Y < bounds[k + 1])

        else:

            mask = (Y >= bounds[k]) & (Y <= bounds[k + 1])

        idx[mask] = k

    # ---------------------------------------------------------
    # Group centers
    # ---------------------------------------------------------

    centers = np.zeros(n_eff)

    for k in range(n_eff):

        centers[k] = np.mean(Y[idx == k])

    return idx, bounds, centers, n_eff

def rsa_pairwise_distances(X, Y, ngroup=10):
    """
    Вычисляет матрицу попарных KS-расстояний (MVD).
    """
    X = np.asarray(X)
    Y = np.asarray(Y).flatten()
    N, M = X.shape
    
    idx, _, _, ngroup_eff = split_sample(Y, ngroup)
    
    if ngroup_eff < ngroup:
        warn(f"Использовано {ngroup_eff} групп вместо {ngroup} из-за повторяющихся значений Y.")
        
    n_pairs = int(ngroup_eff * (ngroup_eff - 1) / 2)

    # Матрица всех попарных KS-расстояний
    ks_matrix = np.full((n_pairs, M), np.nan)

    for i in range(M):

        xx = np.unique(np.sort(X[:, i]))
        L = len(xx)

        CDF_ = np.full((L, ngroup_eff), np.nan)

        for j in range(ngroup_eff):
            CDF_[:, j] = empiricalcdf(X[idx == j, i], xx)

        count = 0

        for j in range(ngroup_eff):
            for k in range(j + 1, ngroup_eff):

                ks_matrix[count, i] = np.max(
                    np.abs(CDF_[:, j] - CDF_[:, k])
                    )

                count += 1

    # Настоящий Mean Vertical Distance
    mvd = np.mean(ks_matrix, axis=0)

    # Разброс KS
    spread = np.max(ks_matrix, axis=0) - np.min(ks_matrix, axis=0)

    return ks_matrix, mvd, spread

import numpy as np


def standardize_data(input_data):
    """
    Standardizes data by subtracting the mean and dividing by the standard deviation
    across axis=0 (rows). The resulting data has 0 mean and standard deviation of 1 (unit variance).
    :param input_data: Data to be standardized.
    :return: Standardized data.
    """
    m = np.mean(input_data, axis=0)
    s = np.std(input_data, axis=0, ddof=1)
    input_data = (input_data - m) / s
    return input_data


def normalize_data(input_data):
    """
    Normalizes data to a [0, 1] range.
    :param input_data: Data to be normalized.
    :return: Normalized data.
    """
    min_val = np.min(input_data, axis=0)
    max_val = np.max(input_data, axis=0)
    input_data = (input_data - min_val) / (max_val - min_val)
    return input_data


def find_correlated(corr, corr_with_target, threshold=0.65):
    """
    Identifies pairwise correlated features and chooses the better one (less important one)
    to be dropped from the model, based on the correlation with the target.
    :param corr: Pairwise correlation matrix of all the features in the model (except the target).
    :param corr_with_target: Matrix of correlations of each feature with the target.
    :param threshold: Variables are considered highly correlated if their correlation > threshold.
    """
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    correlated = set()
    for col in upper.columns:
        correlated_vars = upper[upper[col] > threshold].index.values
        if col in correlated:
            continue
        for var in correlated_vars:
            if var in correlated:
                continue
            if corr_with_target.loc[var][0] > corr_with_target.loc[col][0]:
                correlated.add(col)
                corr_with_target.loc[var][0] = 1
                corr_with_target.loc[col][0] = -1
                break
            else:
                correlated.add(var)
                corr_with_target.loc[col][0] = 1
                corr_with_target.loc[var][0] = -1
    return correlated

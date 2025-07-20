import numpy as np
from scipy.stats import sem, t

def mean_margin(series, confidence=0.95):
    """Return mean ± margin of error for a pandas Series."""
    n = len(series)
    mean = np.mean(series)
    se = sem(series)
    margin = se * t.ppf((1 + confidence) / 2., n - 1)
    return f"{mean:.4f} ± {margin:.4f}"

def mean_margin_by_group(df, group_col, confidence=0.95):
    """
    Group a DataFrame by `group_col` and return a new DataFrame with
    mean ± margin of error for all other numeric columns.
    
    Parameters:
    - df: input DataFrame
    - group_col: column to group by
    - confidence: confidence level for the margin of error
    
    Returns:
    - DataFrame with group column and formatted mean ± margin strings
    """
    grouped = df.groupby(group_col)
    
    result = grouped.agg({
        col: lambda x: mean_margin(x, confidence)
        for col in df.columns if col != group_col
    })
    
    result.reset_index(inplace=True)
    return result

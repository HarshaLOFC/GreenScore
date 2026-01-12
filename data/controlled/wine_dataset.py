"""
wine_dataset.py
---------------
Controlled dataset loader for the GreenScore project.

This module loads the Wine dataset from scikit-learn.
The dataset is:
- Fully cleaned
- Numerical
- Multi-class classification

Responsibilities:
1. Load dataset
2. Return features and target separately

NOTE:
- No preprocessing here
- No train/test split here
"""

import pandas as pd
from sklearn.datasets import load_wine


# -------------------------------------------------
# Load Wine Dataset
# -------------------------------------------------
def load_wine_dataset():
    """
    Loads the Wine dataset as a pandas DataFrame.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (independent variables)
    y : pd.Series
        Target labels (dependent variable)
    """

    # Load dataset from scikit-learn
    wine = load_wine(as_frame=True)

    # Full dataframe including target
    df = wine.frame

    # Separate features and target
    X = df.drop(columns=["target"])
    y = df["target"]

    return X, y

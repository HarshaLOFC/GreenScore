def load_custom_dataset(df, target_column):
    """
    Validates and splits a custom dataset for classification.
    """

    if target_column not in df.columns:
        raise ValueError("Selected target column does not exist.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    if X.isnull().any().any() or y.isnull().any():
        raise ValueError("Dataset contains missing values.")

    if not all(dtype.kind in "iufc" for dtype in X.dtypes):
        raise ValueError("All feature columns must be numeric.")

    if y.nunique() < 2:
        raise ValueError("Target column must have at least 2 classes.")

    return X, y

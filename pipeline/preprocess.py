"""
preprocess.py
-------------
Data preprocessing utilities for the GreenScore pipeline.

Responsibilities:
1. Train-test split
2. Feature scaling
3. Return clean data for model training

NOTE:
- Works only for classification tasks (for now)
- Keeps preprocessing logic centralized and reusable
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# -------------------------------------------------
# Preprocess Data
# -------------------------------------------------
def preprocess_data(
    X,
    y,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Splits the dataset into train and test sets
    and applies feature scaling.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target labels
    test_size : float, optional
        Fraction of data to use for testing (default = 0.2)
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    X_train_scaled, X_test_scaled, y_train, y_test
    """

    # -------------------------------
    # Train-Test Split
    # -------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # keeps class distribution balanced
    )

    # -------------------------------
    # Feature Scaling
    # -------------------------------
    scaler = StandardScaler()

    # Fit only on training data
    X_train_scaled = scaler.fit_transform(X_train)

    # Transform test data using same scaler
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

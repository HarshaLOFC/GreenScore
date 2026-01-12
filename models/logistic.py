"""
logistic.py
-----------
Logistic Regression model implementation for GreenScore.

Responsibilities:
1. Train Logistic Regression classifier
2. Predict on test data
3. Compute performance metrics
4. Measure training time

NOTE:
- Energy & carbon tracking will be added at pipeline level
- This file should remain model-specific only
"""

import time
from sklearn.linear_model import LogisticRegression
from utils.metrics import compute_classification_metrics


# -------------------------------------------------
# Train Logistic Regression Model
# -------------------------------------------------
def train_logistic_regression(
    X_train,
    y_train,
    X_test,
    y_test
):
    """
    Trains a Logistic Regression classifier and evaluates it.

    Parameters
    ----------
    X_train : array-like
        Scaled training features
    y_train : array-like
        Training labels
    X_test : array-like
        Scaled test features
    y_test : array-like
        Test labels

    Returns
    -------
    dict
        Dictionary containing model name, accuracy, f1-score, and training time
    """

    # -------------------------------
    # Initialize Model
    # -------------------------------
    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        n_jobs=-1,
        random_state=42
    )

    # -------------------------------
    # Train Model (Time Tracking)
    # -------------------------------
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    # -------------------------------
    # Predictions
    # -------------------------------
    y_pred = model.predict(X_test)

    # -------------------------------
    # Compute Metrics
    # -------------------------------
    metrics = compute_classification_metrics(y_test, y_pred)

    # -------------------------------
    # Collect Results
    # -------------------------------
    results = {
        "Model": "Logistic Regression",
        "Accuracy": metrics["accuracy"],
        "F1-score": metrics["f1_score"],
        "Time (s)": training_time
    }

    return results

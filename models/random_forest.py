"""
random_forest.py
----------------
Random Forest model implementation for the GreenScore project.

Responsibilities:
1. Train Random Forest classifier
2. Predict on test data
3. Compute performance metrics
4. Measure training time

NOTE:
- Random Forest is more computationally expensive than Logistic Regression
- This makes it ideal for GreenScore comparison
"""

import time
from sklearn.ensemble import RandomForestClassifier
from utils.metrics import compute_classification_metrics


# -------------------------------------------------
# Train Random Forest Model
# -------------------------------------------------
def train_random_forest(
    X_train,
    y_train,
    X_test,
    y_test
):
    """
    Trains a Random Forest classifier and evaluates it.

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
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=None,
        random_state=42,
        n_jobs=-1
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
        "Model": "Random Forest",
        "Accuracy": metrics["accuracy"],
        "F1-score": metrics["f1_score"],
        "Time (s)": training_time
    }

    return results

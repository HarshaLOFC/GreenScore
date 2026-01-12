"""
mlp.py
------
Multi-Layer Perceptron (Neural Network) model implementation
for the GreenScore project.

Responsibilities:
1. Train an MLP classifier
2. Predict on test data
3. Compute performance metrics
4. Measure training time

NOTE:
- MLP represents deep learning in this project
- Typically more computationally expensive
"""

import time
from sklearn.neural_network import MLPClassifier
from utils.metrics import compute_classification_metrics


# -------------------------------------------------
# Train MLP Model
# -------------------------------------------------
def train_mlp(
    X_train,
    y_train,
    X_test,
    y_test
):
    """
    Trains an MLP classifier and evaluates it.

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
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=500,
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
        "Model": "Neural Network (MLP)",
        "Accuracy": metrics["accuracy"],
        "F1-score": metrics["f1_score"],
        "Time (s)": training_time
    }

    return results

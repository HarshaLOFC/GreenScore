import os
import pandas as pd
from codecarbon import EmissionsTracker

from data.controlled.wine_dataset import load_wine_dataset
from data.custom_dataset import load_custom_dataset
from pipeline.preprocess import preprocess_data

from models.logistic import train_logistic_regression
from models.random_forest import train_random_forest
from models.mlp import train_mlp

from utils.metrics import compute_greenscore


def run_pipeline(
    dataset_mode,
    selected_models,
    weights,
    uploaded_file=None,
    target_column=None,
):

    os.makedirs("evaluation", exist_ok=True)

    results_path = "evaluation/results.csv"
    emissions_path = "evaluation/emissions.csv"

    if os.path.exists(results_path):
        os.remove(results_path)
    if os.path.exists(emissions_path):
        os.remove(emissions_path)

    # -------------------------------
    # Load dataset
    # -------------------------------
    if dataset_mode == "Controlled Mode (Built-in)":
        X, y = load_wine_dataset()

    elif dataset_mode == "Custom Dataset":
        if uploaded_file is None or target_column is None:
            raise ValueError("Custom dataset and target column must be provided.")

        custom_df = uploaded_file
        X, y = load_custom_dataset(custom_df, target_column)

    else:
        raise ValueError("Invalid dataset mode.")

    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    model_runners = [
        ("Logistic Regression", train_logistic_regression),
        ("Random Forest", train_random_forest),
        ("Neural Network (MLP)", train_mlp),
    ]

    results = []

    for model_name, train_fn in model_runners:

        tracker = EmissionsTracker(
            project_name="GreenScore",
            output_dir="evaluation",
            log_level="error",
        )

        tracker.start()

        model_result = train_fn(
            X_train, y_train, X_test, y_test
        )

        emissions_kg = tracker.stop()

        if os.path.exists(emissions_path):
            emissions_df = pd.read_csv(emissions_path)
            energy_kwh = emissions_df["energy_consumed"].iloc[-1]
        else:
            energy_kwh = 0.0

        model_result["Model"] = model_name
        model_result["Energy (kWh)"] = energy_kwh
        model_result["CO2 (kg)"] = emissions_kg
        model_result["CO2 (tons)"] = emissions_kg / 1000

        results.append(model_result)

    results_df = pd.DataFrame(results)
    results_df = compute_greenscore(results_df, weights)
    results_df.to_csv(results_path, index=False)

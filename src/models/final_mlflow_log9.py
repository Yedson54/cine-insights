from typing import Any

import pandas as pd
import mlflow


def mlflow_fun(
    gmodel: Any, experiment_name: str, num_appli: int, evaluation_data: Any
) -> None:
    """
    Function to perform model training, evaluation, and logging using MLflow.

    Args:
        gmodel (Any): The model object with train_model, evaluate_model, and predict_ methods.
        experiment_name (str): The name of the MLflow experiment.
        num_appli (int): The number of applications.
        evaluation_data (Any): The evaluation data to make predictions on.

    Returns:
        None
    """
    mlflow.set_experiment(experiment_name=experiment_name)
    grid_params = gmodel.param_Combination

    for idx, params in enumerate(grid_params):
        run_name = f"run_{idx}"
        with mlflow.start_run(run_name=run_name):
            # Log hyperparameters
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)

            # Train the model
            gmodel.train_model(params=params)

            # Log fit metrics
            metrics = gmodel.evaluate_model()
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            # Log model as an artifact
            model_artifact_path = f"model_{run_name}.pkl"
            gmodel.save_best_model(model_artifact_path)
            mlflow.log_artifact(model_artifact_path, artifact_path="models")

            # Make predictions on evaluation data
            predictions = gmodel.predict_(evaluation_data)

            # Save predictions as an artifact
            predictions_df = pd.DataFrame({"Prediction": predictions})
            predictions_file_path = f"predictions_{run_name}.csv"
            predictions_df.to_csv(predictions_file_path, index=False)
            mlflow.log_artifact(predictions_file_path, artifact_path="predictions")

            mlflow.log_table(
                data=predictions_df, artifact_file=run_name + "predictions_df.json"
            )
            # Log training data URL
            mlflow.log_param("num_appli", num_appli)

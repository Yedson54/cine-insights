import mlflow
import pandas as pd
filename=".csv"
path_file="/home/onyxia/work/cine-insights/src/data/"
def mlflow_fun(gmodel, experiment_name, num_appli, evaluation_data):
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

            mlflow.log_table(data=predictions_df, artifact_file=run_name+"predictions_df.json")
            # Log training data URL
            mlflow.log_param("num_appli", num_appli)

import mlflow
import os


import pandas as pd

def filter_params_indices(series):
    # Utiliser une expression régulière pour filtrer les indices commençant par "params."
    filtered_indices = series.index[series.index.str.startswith("params.")]
    # Sélectionner les lignes de la série correspondant aux indices filtrés.
    filtered_series = series.loc[filtered_indices]
    return filtered_series


def get_best_model(experiment_name):
    # Récupérer l'ID de l'expérience à partir de son nom
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment:
        experiment_id = experiment.experiment_id
    else:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    # Récupérer toutes les exécutions dans cette expérience
    runs = mlflow.search_runs(experiment_ids=[experiment_id])

    # Trouver la meilleure exécution en fonction d'une métrique donnée (par exemple, RMSLE)
    best_run = runs.loc[runs['metrics.Test RMSLE'].idxmin()]

    # Afficher les métadonnées de la meilleure exécution
    print("Best run:")
    print(best_run)

    # Récupérer les informations sur les hyperparamètres et l'emplacement de l'artefact
    best_params = filter_params_indices(best_run)
    model_artifact_uri = best_run["artifact_uri"] + "/model"

    return best_params, model_artifact_uri

def load_model(artifact_uri):
    # Charger le modèle à partir de l'emplacement de l'artefact
    model_path = os.path.join(artifact_uri, "model.pkl")
    return mlflow.sklearn.load_model(model_path)

if __name__ == "__main__":
    # Nom de l'expérience MLflow
    experiment_name = "Experiment2:save only paramater and metrics"

    # Récupérer les meilleurs hyperparamètres et l'emplacement du meilleur modèle
    best_params, model_artifact_uri = get_best_model(experiment_name)

    # Charger le meilleur modèle
    best_model = load_model(model_artifact_uri)

    # Utiliser le meilleur modèle pour faire des prédictions, etc.
    # Par exemple :
    # y_pred = best_model.predict(X_test)
    # etc.

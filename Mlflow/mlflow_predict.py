








import sys
sys.path.append('/home/onyxia/work/cine-insights/src/models')
sys.path.append('/home/onyxia/work/cine-insights/src/data')
sys.path.append('/home/onyxia/work/cine-insights/src/features')
sys.path.append('/home/onyxia/work/cine-insights/src/models')
sys.path.append('/home/onyxia/work/cine-insights/models') #/home/onyxia/work/cine-insights/src/Mlflow_
sys.path.append('/home/onyxia/work/cine-insights/src/Mlflow_') #

import mlflow
import pandas as pd
import Import_data
from Import_data import *
import Preprocess_predicted_data
from Preprocess_predicted_data import *

# data qui ne contient pas la colonne "revenue" et qui aprtagent les même colonnes que le dataset qu'AYMAN me fournira 

filename_="X.csv" 
path_='/home/onyxia/work/cine-insights/src/Mlflow_/'+filename_



data=Import_data.load_data(path_,name="Pred_data")



import Preprocessing_PredictedData
from Preprocess_predicted_data import *

import mlflow.pyfunc

model_name = "Best_model1"
model_version = 1

loaded_model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")


import mlflow
import pickle

# Chemin de l'artefact contenant le transformateur de données dans MLflow
artifact_path = "/home/onyxia/work/cine-insights/notebooks/ct_data_transformer.pkl"
import re

logged_model = 'runs:/0ed2877790af425ca20dce1219a2e288/model_run_51.pkl'

# Utiliser une expression régulière pour extraire l'ID du run
match = re.search(r'runs:/(\w+)/', logged_model)

# l'ID du model : je dois pouvois automatiser ça en utilisant une fonction qui prend disons le nom du run qui est dispo
#dand MLflow et me retourne l'id
run_id="98a7ab75df434e82aaa5ab52e1fe6951"

# Charger le transformateur à partir de MLflow
#with mlflow.start_run(run_id=run_id):
import pickle
# Charger le transformateur de données à partir du fichier pickle
with open(artifact_path, "rb") as f:
    ct_data_transformer = pickle.load(f)

# Transformer les données afin d'avoir les  bonnes colonnes attendus par le modéle
X_transform = Preprocessing_PredictedData.preprocessing_pipeline(data, ct_data_transformer, with_duration=False)
# appliquer l'ensemble de pipeline necessaire comme celle dans l'entrainement
transformed_data = ct_data_transformer.transform(X_transform)
transformed_data=pd.DataFrame(transformed_data,columns=X_transform.columns)




# Utiliser le transformateur de données
#transformed_data = ct_data_transformer.transform(data)



"""
# Predict on a Pandas DataFrame.

loaded_model.predict(data)
"""
def do_():
    return loaded_model.predict(transformed_data)

    

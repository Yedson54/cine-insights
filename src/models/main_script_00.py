import warnings

# Ignorer tous les avertissements
warnings.filterwarnings("ignore")
import sys
sys.path.append('/home/onyxia/work/cine-insights/src/models')
sys.path.append('/home/onyxia/work/cine-insights/src/data')
sys.path.append('/home/onyxia/work/cine-insights/src/features')

import pandas as pd
import joblib
import numpy as np

import Import_data
from Import_data import *

import  Preprocess_trainData  
from Preprocess_trainData import *

import Preprocessing_PredictedData  # requirements train_model_xgboot22a6 and  Build_Feats1a
from  Preprocessing_PredictedData import *  #  Utilisez les fonctions du module file

import Final_Trainer_0
from  Final_Trainer_0 import *


filename="Raw_df_50_ligne.csv"
path_dir="/home/onyxia/work/cine-insights/src/data/"
python_inter_pretor=""


def run_xgreg(path=path_dir+filename):
       raw_df=Import_data.load_data(path,name="Raw_df")
#(X_train,y_train),(X_test,y_test),(X_val,y_val),ct_data_transformer=BON_Predict_model_A210.preprocessing_data(raw_df)
       model=Final_Trainer_0.XGBRegressorWrapper(raw_df,with_duration=False)
       df=model.display_results()

       return df
    

run_xgreg()



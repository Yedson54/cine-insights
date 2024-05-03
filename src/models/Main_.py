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

import train_models
from train_models import *

filename="Raw_df_50_ligne.csv"
path_dir="/home/onyxia/work/cine-insights/src/data/"

# params_=

def run_xgreg(path=path_dir+filename):
       raw_df=Import_data.load_data(path,name="Raw_df")
       model=train_models.XGBRegressorWrapper(raw_df,with_duration=False)
       
       return  print(model.param_Combination[1:3])


run_xgreg()



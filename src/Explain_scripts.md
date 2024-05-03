# 1 Import_data
Ce fichier  prend les données sous formes csv depuis puis les charges dans un data frames.
Il prend comme entrée le chemin où sont stockés les données.
Je sais pas si je doit plutot utiliser le lien des données depuis le systéme de stockage (Minio -fournis par AYAMAN)

# 2 Import_data_by_Park 
Ce contient la même fonction load_data que le fichier Import_data à la seulle différence, il lu les données de formes Parket

# 2 Import_data_by_Park 
Ce contient la même fonction load_data que le fichier Import_data à la seulle différence, il lu les données de formes Parket

# 3  Preprocessing_trainData
ce script contient le pipeline de transformation des données brutes i.e celles lues à partir des deux scripts précédentes
Ce sont les données que le modéle va prendre pour faire ces taches (entrainement,predictions )

# 4 Preprocessing_PredictedData
Ce script prendre des données fournis par nos Import data mais cette fois ci sans la colonne "revenue" (Input variable) puis fait le preprocessing nécessaire avant de faire la prediction

## 5 train_models
Ce script permet de prend en entrée  preprocessing par la pipeline  Preprocessing_trainData et l'ensemble des hyperparametres de notre modele - le learning_rate,subsample...- puis permettra la cross Validations de faire l'entrainement pour chaque combinaison hyperparamétres de nos modéles mais en pratiques l'entraianes ce fera avec Mlfloc
# 6 Main_
Ce script permet de permettait avant de faire l'entrainement et de plotter un dataframe de performance contenant les informations tels que le RSME, le RSLME pour le test et pour le train mais à ce moment où j'écris cette partie  elle permet juste d'afficher  2 deux combinaisons  de nos hyperparamétres. Je dois normalement afficher les résultats comme ce qu'il faisait auparavant mais comme mainteant mes entrainement ne sont  plus fait en local , je ne voyais pas comment le faire. Je viens je viens just d'avoir une idée qui consiste à récupérér à partir de MLFLOW le meilleur modéle et de l'utilise dans le main pour afficher des résulats.
Pour résumer, donc , on peut se dire que le main va à la fin permettre de charger les données depuis le stockage externe faire l'entrainement sur sur plusieur s combinason d'hyperparamétres puis afficher les résualts 
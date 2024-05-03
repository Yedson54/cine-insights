import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler



# Colonnes spécifiques
# Colonnes spécifiques
cols_to_count_values = ["spoken_languages","production_countries","production_companies","Keywords"] # + cats +crew
date_col = "release_date"
Cols_to_Remove = ['Keywords', 'spoken_languages','homepage', 'production_countries','production_companies', 'release_date', 'poster_path', 'id', 'status','imdb_id', 'logRevenue', 'logBudget',"released"]
with_duration = False
log_num_feats = ['budget', 'popularity']
cols_to_binarize = ['homepage','status']
cat_feats = ['has_homepage', 'release_month', 'release_year']
genre_feats = ['genre_Fantasy', 'genre_Action', 'genre_TV_Movie', 'genre_Romance',
               'genre_Western', 'genre_Animation', 'genre_Music', 'genre_Horror',
               'genre_History', 'genre_Mystery', 'genre_Family', 'genre_Drama',
               'genre_Science_Fiction', 'genre_War', 'genre_Adventure',
               'genre_Documentary', 'genre_Thriller', 'genre_Crime', 'genre_Comedy']
count_feats = ['production_countries_count', 'production_companies_count',
               'spoken_languages_count', 'keyword_count', 'Duration']

today = datetime(2024, 4, 14)

def shuffle_split(df, scale=0.25, target="revenue"):
    X = df.drop(target, axis=1)
    y = df[target]

    X_train, X_remind, y_train, y_remind = train_test_split(X, y, train_size=(1-scale), random_state=42)
    X_val, X_test,  y_val, y_test = train_test_split(X_remind, y_remind, test_size=0.8, random_state=42)

    print("(X.shape, y.shape )","\n")  
    print("Pour le train :", X_train.shape, y_train.shape,"\n")
    print("Pour le test :" ,X_test.shape, y_test.shape,"\n")
    print("Pour la validation :", X_val.shape, y_val.shape,"\n")

    return (X_train, y_train), (X_test, y_test), (X_val, y_val)

def change_name(df, old_name="keywords", new_name='Keywords'):
    if old_name in df.columns:
        df = df.rename(columns={old_name: new_name})
    return df


cols_to_drop = ["title","vote_average","vote_count","runtime","adult","backdrop_path","original_title","overview","tagline","original_title"]


def set_cols(df, cols_to_drop = cols_to_drop):
    all_columns = df.columns
    df = df[list(set(all_columns).difference(set(cols_to_drop)))]
    return df


def remove_negative_money(df):
    df_ = df[(df.budget > 0) & (df.revenue > 0)].copy()
    return df_


def fillnan(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            # Remplacer les valeurs manquantes par le mode pour les colonnes catégorielles
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            # Remplacer les valeurs manquantes par la médiane pour les colonnes numériques
            df[col].fillna(df[col].median(), inplace=True)
    return df





def Binarizer(df, cols_to_binarize):
    df_copy = df.copy()
    for col in cols_to_binarize:
        if col=='status':
            num_col_name ="released"
        elif col == "homepage":
            num_col_name = "has_homepage"

        df_copy[num_col_name] = 1
        df_copy.loc[pd.isnull(df_copy[col]), num_col_name] = 0
    df_copy = df_copy.drop(cols_to_binarize,axis=1).copy()
    return df_copy


def split_data_vs_label(df,target='revenue') :
      y = df[target]
      X = df.drop(target, axis=1)
      return X,y



def count_strings(s):
    if pd.isna(s):
        return np.nan
    return len(s.split(','))


def remove_empty_date_line(X_df, Y_df, date_col=date_col):
    X_df, Y_df = X_df[X_df[date_col].notnull()], Y_df[X_df[date_col].notnull()]
    return X_df, Y_df

def yearfix(x):
    r = x[:4]
    return int(r)

def apply_yearfix(df, date_col=date_col, col_name="release_year"):
    df[col_name] = df[date_col].apply(lambda x: yearfix(x))
    return df

def monthfix(x):
    r = x[5:7]
    return int(r)

def apply_monthfix(df, date_col=date_col, col_name="release_month"):
    df[col_name] = df[date_col].apply(lambda x: monthfix(x))
    return df 

def str_to_datetime(str_date, today):
    date_reference = datetime.strptime(str_date, "%Y-%m-%d")
    difference = today - date_reference
    return round(difference.total_seconds() / (3600 * 24), 5)

def add_duration_col(df, with_duration=with_duration, date_col="release_date"):
    if with_duration:
        # Convertir la colonne date_col en chaînes de caractères (str)
        df[date_col] = df[date_col].astype(str)
        today = datetime(2024, 4, 14)
        df["Duration"] = df[date_col].apply(lambda x: str_to_datetime(x, today))
    return df

def model_features(df, cols_to_remove=Cols_to_Remove):
    return list(set(df.columns) - set(cols_to_remove))

class Log1pTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, features=None):
        self.features = features
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.log1p(X)

def log_pipeline():
    log_pipe = Pipeline([
        ('imputer', SimpleImputer(missing_values=np.nan, strategy='median')),
        ('logger', Log1pTransformer()),
        ('scaler', MinMaxScaler())
    ])
    return log_pipe

def num_pipeline():
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(missing_values=np.nan,strategy='median')),
        ('scaler', MinMaxScaler())
    ])
    return num_pipeline

def date_pipeline():
    date_pipeline = Pipeline([
        ('imputer', SimpleImputer(missing_values=np.nan, strategy='median'))
    ])
    return date_pipeline

def cat_pipeline():
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(missing_values=np.nan,strategy='most_frequent')),
    ])
    return cat_pipeline

def encode_pipeline():
    encode_pipeline = Pipeline([
        ('imputer', SimpleImputer(missing_values=np.nan,strategy='most_frequent'))
    ])
    return encode_pipeline

def hash_pipeline():
    hash_pipeline = Pipeline([
        ('imputer', SimpleImputer(missing_values=np.nan,strategy='most_frequent'))
    ])
    return hash_pipeline
    


def apply_count(df, cols_to_count_values):
    df_copy = df.copy()
    for col in cols_to_count_values:
        new_col_name = col + "_count"
        df_copy[new_col_name] = df_copy[col].apply(count_strings)
    df_final = df_copy.drop(cols_to_count_values, axis=1)
    return df_final


def add_gender_cols(df):
    X_all_genres = df['genres'].str.split(', ', expand=True)
    X_all_genres = pd.get_dummies(X_all_genres.apply(lambda x: pd.Series(x).str.strip()), prefix='', prefix_sep='').groupby(level=0, axis=1).sum()
    X_all_genres.columns = ['genre_' + col.replace(' ', '_') for col in X_all_genres.columns]
    df1 = df.drop(["genres"],axis=1).copy() 
    X_all = pd.concat([df1, X_all_genres], axis=1).copy()
    return X_all
    
def genre_column_names(all_cols):
    # Séquence à rechercher
      sequence = 'genre_'

# Filtrer les éléments de la liste qui commencent par la séquence
      genre_column_names = [element for element in all_cols if element.startswith(sequence)]
      return list(genre_column_names)

def preprocessing_pipeline(df,with_duration=with_duration):
    # Appliquer les transformations sur df
    big_df = change_name(df).copy()
    big_df = set_cols(big_df)
    big_df = remove_negative_money(big_df)
    df_with_count_col = apply_count(big_df, cols_to_count_values=cols_to_count_values)
    big_df = Binarizer(df_with_count_col, cols_to_binarize)  # Si Binarizer est une fonction définie ailleurs
    big_df = add_gender_cols(big_df).copy()  # Si add_gender_cols est une fonction définie ailleurs
    big_df, Y = split_data_vs_label(big_df)
    big_df, Y = remove_empty_date_line(big_df, Y)  # remplacer le deuxieme argument par y et recuperer y_big
    big_df=fillnan(big_df)
    big_df = add_duration_col(big_df, with_duration=with_duration)
    big_df = apply_monthfix(big_df)
    big_df = apply_yearfix(big_df)
    big_df = big_df.drop(date_col, axis=1)
    genre_feats = genre_column_names(big_df.columns)  # Si genre_column_names est une fonction définie ailleurs

    # Définir les colonnes à utiliser dans le ColumnTransformer
    # cat_feats = ['has_homepage']
    count_feats = ['spoken_languages_count', 'production_countries_count',
       'production_companies_count', 'Keywords_count']
    if with_duration==True:
        num_feats = ["Duration"] # non requis, à explorer ultérieurement
        names = log_num_feats + cat_feats + genre_feats + count_feats + num_feats + ['has_homepage', 'released'] 
    else :
        names = log_num_feats + cat_feats + genre_feats + count_feats + ['has_homepage', 'released'] 

    X = big_df[names]  # Pas besoin de copier car vous l'avez déjà fait lors de la transformation
    X = X.loc[:, ~X.columns.duplicated()]

    # Définir les transformateurs
    log_pipe = log_pipeline()
    cat_pipe = cat_pipeline()
    hash_pipe = hash_pipeline()

    # Appliquer le ColumnTransformer
    ct = ColumnTransformer([
        ('log_num_feats', log_pipe, log_num_feats),
        ('cat_feats', cat_pipe, cat_feats),
        ('hash_feats', hash_pipe, genre_feats+count_feats) #+ count_feats)
    ], remainder='passthrough').fit(X) 
    
    transformed_columns = log_num_feats + cat_feats + genre_feats + count_feats 

    # Obtenez les noms de colonnes transformées

    # Obtenez les noms de colonnes non transformées
    remaining_columns = [col for col in X.columns if col not in transformed_columns]

    # Combinez les deux listes de noms de colonnes
    all_columns = transformed_columns + remaining_columns

    transformed_data = ct.transform(X)

    # Créez un DataFrame avec les données transformées et les noms de colonnes
    transformed_df = pd.DataFrame(transformed_data, columns=all_columns)

    return transformed_df, Y,ct
    

def shuffle_split(X,y,scale=0.25):   

    X_train, X_remind, y_train, y_remind = train_test_split(X, y, train_size=(1-scale), random_state=42)
    X_val, X_test,  y_val, y_test = train_test_split(X_remind, y_remind, test_size=0.8, random_state=42)
    # 0.8 = 80% des donnes restant por le jeu de test et les 20% pour la validation

    print("(x.shape, y.shape )","\n")
    print ( " Pour le train :", X_train.shape, y_train.shape,"\n" )
    print ( " Pour le test :" ,X_test.shape, y_test.shape,"\n" )
    print ( " Pour la validation :", X_val.shape, y_val.shape,"\n" )

    return (X_train, y_train), ( X_test, y_test),( X_val, y_val)

def preprocessing_data(raw_data,with_duration=with_duration):
    X ,Y,ct = preprocessing_pipeline(raw_data,with_duration)
    (X_train, y_train), ( X_test, y_test),( X_val, y_val)=shuffle_split(X,Y)
    return (X_train, y_train), ( X_test, y_test),( X_val, y_val),ct




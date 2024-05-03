"""Module to preprocess the data for the prediction tasks.
"""

from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd


# Colonnes spécifiques
cols_to_count_values = [
    "spoken_languages",
    "production_countries",
    "production_companies",
    "Keywords",
]  # + cats +crew
DATE_COL = "release_date"
cols_to_drop = [
    "Keywords",
    "spoken_languages",
    "homepage",
    "production_countries",
    "production_companies",
    "release_date",
    "poster_path",
    "id",
    "status",
    "imdb_id",
    "logRevenue",
    "logBudget",
    "released",
]
WITH_DURATION = False
log_num_feats = ["budget", "popularity"]
cols_to_binarize = ["homepage", "status"]
cat_feats = ["has_homepage", "release_month", "release_year"]
genre_feats = [
    "genre_Fantasy",
    "genre_Action",
    "genre_TV_Movie",
    "genre_Romance",
    "genre_Western",
    "genre_Animation",
    "genre_Music",
    "genre_Horror",
    "genre_History",
    "genre_Mystery",
    "genre_Family",
    "genre_Drama",
    "genre_Science_Fiction",
    "genre_War",
    "genre_Adventure",
    "genre_Documentary",
    "genre_Thriller",
    "genre_Crime",
    "genre_Comedy",
]
count_feats = [
    "production_countries_count",
    "production_companies_count",
    "spoken_languages_count",
    "keyword_count",
    "Duration",
]

today = datetime(2024, 4, 14)


def change_name(
    df: pd.DataFrame, old_name: str = "keywords", new_name: str = "Keywords"
) -> pd.DataFrame:
    """Change the column name in the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        old_name (str, optional): Old column name. Defaults to "keywords".
        new_name (str, optional): New column name. Defaults to "Keywords".

    Returns:
        pd.DataFrame: DataFrame with the column name changed.
    """
    if old_name in df.columns:
        df = df.rename(columns={old_name: new_name})
    return df


def set_cols(df: pd.DataFrame, cols_to_drop: List[str] = cols_to_drop) -> pd.DataFrame:
    """Set the columns of the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        cols_to_drop (List[str], optional): Columns to drop. Defaults to cols_to_drop.

    Returns:
        pd.DataFrame: DataFrame with the specified columns dropped.
    """
    all_columns = df.columns
    df = df[list(set(all_columns).difference(set(cols_to_drop)))]
    return df


def remove_negative_money(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with negative budget values from the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with rows having negative budget values removed.
    """
    df_ = df[df.budget > 0].copy()
    return df_


def fillnan(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values in the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with missing values filled.
    """
    for col in df.columns:
        if df[col].dtype == "object":
            # Replace missing values with mode for categorical columns
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            # Replace missing values with median for numerical columns
            df[col].fillna(df[col].median(), inplace=True)
    return df


def Binarizer(
    df: pd.DataFrame, cols_to_binarize: List[str] = cols_to_binarize
) -> pd.DataFrame:
    """Binarize the specified columns in the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        cols_to_binarize (List[str]): Columns to binarize.

    Returns:
        pd.DataFrame: DataFrame with specified columns binarized.
    """
    df_copy = df.copy()
    for col in cols_to_binarize:
        if col == "status":
            num_col_name = "released"
        elif col == "homepage":
            num_col_name = "has_homepage"

        df_copy[num_col_name] = 1
        df_copy.loc[pd.isnull(df_copy[col]), num_col_name] = 0
    df_copy = df_copy.drop(cols_to_binarize, axis=1).copy()
    return df_copy


def split_data_vs_label(
    df: pd.DataFrame, target: str = "revenue"
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split the DataFrame into features and target variables.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target (str, optional): Target variable. Defaults to "revenue".

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features DataFrame and target Series.
    """
    y = df[target]
    X = df.drop(target, axis=1)
    return X, y


def count_strings(s: str) -> int:
    """Count the number of strings in a comma-separated string.

    Args:
        s (str): Input string.

    Returns:
        int: Number of strings.
    """
    if pd.isna(s):
        return np.nan
    return len(s.split(","))


def remove_empty_date_line(
    X_df: pd.DataFrame, date_col: str = DATE_COL
) -> pd.DataFrame:
    """Remove rows with empty date values from the DataFrame.

    Args:
        X_df (pd.DataFrame): Input DataFrame.
        date_col (str, optional): Date column. Defaults to date_col.

    Returns:
        pd.DataFrame: DataFrame with rows having empty date values removed.
    """
    X_df = X_df[X_df[date_col].notnull()]
    return X_df


def yearfix(x: str) -> int:
    """Extract the year from a date string.

    Args:
        x (str): Date string.

    Returns:
        int: Year.
    """
    r = x[:4]
    return int(r)


def apply_yearfix(
    df: pd.DataFrame, date_col: str = DATE_COL, col_name: str = "release_year"
) -> pd.DataFrame:
    """Apply the yearfix function to a date column and create a new column with the year.

    Args:
        df (pd.DataFrame): Input DataFrame.
        date_col (str, optional): Date column. Defaults to date_col.
        col_name (str, optional): New column name. Defaults to "release_year".

    Returns:
        pd.DataFrame: DataFrame with the new column added.
    """
    df[col_name] = df[date_col].apply(yearfix)
    return df


def monthfix(x: str) -> int:
    """Extract the month from a date string.

    Args:
        x (str): Date string.

    Returns:
        int: Month.
    """
    r = x[5:7]
    return int(r)


def apply_monthfix(
    df: pd.DataFrame, date_col: str = DATE_COL, col_name: str = "release_month"
) -> pd.DataFrame:
    """Apply the monthfix function to a date column and create a new column with the month.

    Args:
        df (pd.DataFrame): Input DataFrame.
        date_col (str, optional): Date column. Defaults to date_col.
        col_name (str, optional): New column name. Defaults to "release_month".

    Returns:
        pd.DataFrame: DataFrame with the new column added.
    """
    df[col_name] = df[date_col].apply(monthfix)
    return df


def str_to_datetime(str_date: str, today: datetime) -> float:
    """Convert a string date to datetime and calculate the difference in days from a reference date.

    Args:
        str_date (str): Input string date.
        today (datetime): Reference date.

    Returns:
        float: Difference in days.
    """
    date_reference = datetime.strptime(str_date, "%Y-%m-%d")
    difference = today - date_reference
    return round(difference.total_seconds() / (3600 * 24), 5)


def add_duration_col(
    df: pd.DataFrame,
    with_duration: bool = WITH_DURATION,
    date_col: str = "release_date",
) -> pd.DataFrame:
    """Add a duration column to the DataFrame based on the difference in days from a reference date.

    Args:
        df (pd.DataFrame): Input DataFrame.
        with_duration (bool, optional): Flag to indicate whether to add the duration column. Defaults to with_duration.
        date_col (str, optional): Date column. Defaults to "release_date".

    Returns:
        pd.DataFrame: DataFrame with the duration column added.
    """
    if with_duration == True:
        # Convert the date_col column to strings (str)
        df[date_col] = df[date_col].astype(str)
        today = datetime(2024, 4, 14)
        df["Duration"] = df[date_col].apply(lambda x: str_to_datetime(x, today))
    return df


def model_features(
    df: pd.DataFrame, cols_to_remove: List[str] = cols_to_drop
) -> List[str]:
    """Get the model features by removing specified columns from the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        cols_to_remove (List[str], optional): Columns to remove. Defaults to Cols_to_Remove.

    Returns:
        List[str]: List of model features.
    """
    return list(set(df.columns) - set(cols_to_remove))


def apply_count(
    df: pd.DataFrame, cols_to_count_values: List[str] = cols_to_count_values
) -> pd.DataFrame:
    """Apply the count_strings function to specified columns in the DataFrame and create new count columns.

    Args:
        df (pd.DataFrame): Input DataFrame.
        cols_to_count_values (List[str]): Columns to apply count_strings function.

    Returns:
        pd.DataFrame: DataFrame with new count columns added.
    """
    df_copy = df.copy()
    for col in cols_to_count_values:
        new_col_name = col + "_count"
        df_copy[new_col_name] = df_copy[col].apply(count_strings)
    df_final = df_copy.drop(cols_to_count_values, axis=1)
    return df_final


def add_gender_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Add genre columns to the DataFrame based on the "genres" column.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with genre columns added.
    """
    X_all_genres = df["genres"].str.split(", ", expand=True)
    X_all_genres = (
        pd.get_dummies(
            X_all_genres.apply(lambda x: pd.Series(x).str.strip()),
            prefix="",
            prefix_sep="",
        )
        .groupby(level=0, axis=1)
        .sum()
    )
    X_all_genres.columns = [
        "genre_" + col.replace(" ", "_") for col in X_all_genres.columns
    ]
    df1 = df.drop(["genres"], axis=1).copy()
    X_all = pd.concat([df1, X_all_genres], axis=1).copy()
    return X_all


def genre_column_names(all_cols: List[str]) -> List[str]:
    """Get the genre column names from a list of all column names.

    Args:
        all_cols (List[str]): List of all column names.

    Returns:
        List[str]: List of genre column names.
    """
    # Sequence to search for
    sequence = "genre_"

    # Filter elements in the list that start with the sequence
    genre_colnames = [
        element for element in all_cols if element.startswith(sequence)
    ]
    return list(genre_colnames)


def preprocessing_pipeline(
    df: pd.DataFrame, ct, with_duration: bool = WITH_DURATION, date_col: str = DATE_COL
) -> pd.DataFrame:
    """Apply a series of preprocessing transformations to the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        ct: ColumnTransformer object.
        with_duration (bool, optional): Flag to indicate whether to include duration column. Defaults to with_duration.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    # Apply transformations on df
    big_df = change_name(df).copy()
    big_df = set_cols(big_df)
    big_df = remove_negative_money(big_df)
    df_with_count_col = apply_count(big_df, cols_to_count_values=cols_to_count_values)
    big_df = Binarizer(df_with_count_col, cols_to_binarize)
    big_df = add_gender_cols(big_df).copy()
    big_df = remove_empty_date_line(big_df)
    big_df = fillnan(big_df)
    big_df = add_duration_col(big_df, with_duration=with_duration)
    big_df = apply_monthfix(big_df)
    big_df = apply_yearfix(big_df)
    big_df = big_df.drop(date_col, axis=1)
    genre_features = genre_column_names(big_df.columns)

    # Define columns to use in ColumnTransformer
    if with_duration:
        num_feats = ["Duration"]
        names = (
            log_num_feats
            + cat_feats
            + genre_features
            + count_feats
            + num_feats
            + ["has_homepage", "released"]
        )
    else:
        names = (
            log_num_feats
            + cat_feats
            + genre_feats
            + count_feats
            + ["has_homepage", "released"]
        )

    X = big_df[names]
    X = X.loc[:, ~X.columns.duplicated()]

    transformed_columns = log_num_feats + cat_feats + genre_feats + count_feats
    remaining_columns = [col for col in X.columns if col not in transformed_columns]
    all_columns = transformed_columns + remaining_columns

    transformed_data = ct.fit_transform(X)
    transformed_df = pd.DataFrame(transformed_data, columns=all_columns)

    return transformed_df

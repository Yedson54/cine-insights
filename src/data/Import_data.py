"""Import data module.
"""

# pylint: disable=locally-disabled, fixme, invalid-name, too-many-arguments, too-many-instance-attributes

import pandas as pd
from pyarrow.parquet import ParquetFile
from IPython import display

def load_data(in_path, name, n_display=1, show_info=False, sep=",", nrows=720000):
    """
    Load data from either a CSV or Parquet file based on the file extension.

    Args:
        in_path (str): The path to the input file.
        name (str): The name of the dataset.
        n_display (int, optional): The number of rows to display. Defaults to 1.
        show_info (bool, optional): Whether to display information about the DataFrame. Defaults to False.
        sep (str, optional): The delimiter used in the CSV file. Defaults to ",".
        nrows (int, optional): The number of rows to read from the CSV file. Defaults to 720000.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    file_extension = in_path.split('.')[-1]

    if file_extension == 'csv':
        df = load_data_csv(in_path, name, n_display, show_info, sep, nrows)
    elif file_extension == 'parquet':
        df = load_data_parquet(in_path, name, n_display, show_info)
    else:
        raise ValueError("Unsupported file format. Only CSV and Parquet are supported.")

    return df

def load_data_csv(in_path, name, n_display=1, show_info=False, sep=",", nrows=720000):
    """
    Load data from a CSV file.

    Args:
        in_path (str): The path to the input CSV file.
        name (str): The name of the dataset.
        n_display (int, optional): The number of rows to display. Defaults to 1.
        show_info (bool, optional): Whether to display information about the DataFrame. Defaults to False.
        sep (str, optional): The delimiter used in the CSV file. Defaults to ",".
        nrows (int, optional): The number of rows to read from the CSV file. Defaults to 720000.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    df = pd.read_csv(in_path, sep=sep, nrows=nrows)
    print(f"{name}: shape is {df.shape}")
    df = df.rename(columns={"keywords": "Keywords"})

    if show_info:
        print(df.info())

    if n_display > 0:
        display.display(df.head(n_display))

    return df

def load_data_parquet(in_path, name, n_display=1, show_info=False):
    """
    Load data from a Parquet file.

    Args:
        in_path (str): The path to the input Parquet file.
        name (str): The name of the dataset.
        n_display (int, optional): The number of rows to display. Defaults to 1.
        show_info (bool, optional): Whether to display information about the DataFrame. Defaults to False.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    pf = ParquetFile(in_path)
    df = pf.read().to_pandas()
    print(f"{name}: shape is {df.shape}")
    df = df.rename(columns={"keywords": "Keywords"})

    if show_info:
        print(df.info())

    if n_display > 0:
        display.display(df.head(n_display))

    return df

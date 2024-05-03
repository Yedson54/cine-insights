"""Import data module.
"""

# pylint: disable=locally-disabled, fixme, invalid-name, too-many-arguments, too-many-instance-attributes

import pandas as pd
import IPython.display as display


def load_data(in_path, name, n_display=1, show_info=False, nrows=None):
    df = pd.read_parquet(in_path, nrows=nrows)
    print(f"{name}: shape is {df.shape}")
    df = df.rename(columns={"keywords": "Keywords"})

    if show_info:
        print(df.info())

    if n_display > 0:
        display.display(df.head(n_display))

    return df


def load_data_csv(
    in_path: str,
    name: str,
    n_display: int = 1,
    show_info: bool = False,
    sep: str = ",",
    nrows: int = 720000,
) -> pd.DataFrame:
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
        df.head(n_display)

    return df

import pandas as pd
import IPython.display as display

def load_data(in_path, name, n_display=1, show_info=False, nrows=None):
    df = pd.read_parquet(in_path, nrows=nrows)
    print(f"{name}: shape is {df.shape}")
    df = df.rename(columns={'keywords': 'Keywords'})

    if show_info:
        print(df.info())
    
    if n_display > 0:
        display.display(df.head(n_display))
        
    return df

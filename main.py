"""Main script to run the model.
"""

# pylint: disable=locally-disabled, multiple-statements, fixme, invalid-name, too-many-arguments, too-many-instance-attributes

from typing import Optional

from src.data import import_data
from src.models import train_models


if __name__ == "__main__":
    filename = "raw_df_50_ligne.csv"
    path_dir = "src/data/"

    def run_xgreg(path: Optional[str] = path_dir + filename) -> None:
        """Run the XGBoost Regressor model.

        Args:
            path (str, optional): Path to the data file. 
                Defaults to path_dir + filename.
        """
        raw_df = import_data.load_data(path, name="Raw_df")
        model = train_models.XGBRegressorWrapper(raw_df, with_duration=False)

        print(model.param_Combination[1:3])

    run_xgreg()

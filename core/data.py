import pandas as pd


def save_data(data: pd.DataFrame, path: str, overwrite=True):
    data.to_csv(
        path,
        mode="w" if overwrite else "a",
        header=overwrite,
    )


def load_data(data_path: str) -> pd.DataFrame:
    return pd.read_csv(data_path, header=[0, 1], skipinitialspace=True, index_col=0)

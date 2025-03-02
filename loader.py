import pandas as pd
from joblib.numpy_pickle import Path
from pandas.io.parsers.base_parser import DataFrame

# Return a list of dataframes where each entry is a patient
def loader(path: str) -> list[DataFrame]:
    training_setA = sorted(Path(path).glob("*.psv"))
    dataframes = []
    for file in training_setA:
        df = pd.read_csv(file, delimiter="|")
        dataframes.append(df)
    return dataframes

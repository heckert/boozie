import io
import requests
import tempfile
import zipfile

from pathlib import Path
from typing import Tuple

import pandas as pd
from omegaconf import OmegaConf


def load_wine() -> pd.DataFrame:

    url = "https://archive.ics.uci.edu/static/public/186/wine+quality.zip"

    response = requests.get(url)

    if response.status_code == 200:

        with tempfile.TemporaryDirectory() as tmpdirname:

            zip_file = zipfile.ZipFile(io.BytesIO(response.content))

            zip_file.extractall(tmpdirname)
            zip_file.close()

            red = pd.read_csv(
                Path(tmpdirname) / "winequality-red.csv", sep=";"
            )
            white = pd.read_csv(
                Path(tmpdirname) / "winequality-white.csv", sep=";"
            )

        df = pd.concat([red, white], ignore_index=True)

    else:
        response.raise_for_status()

    return df


def extract_samples(
    df: pd.DataFrame, samples: dict, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    df = df.copy()
    samples_df = pd.DataFrame()

    for name, score in samples.items():
        subset = df[df["quality"] == score]

        record = (subset.sample(random_state=random_state)
                        .assign(name=name))
        df = df.drop(record.index)
        samples_df = pd.concat([samples_df, record])

    return samples_df, df


if __name__ == "__main__":

    file_path = Path(__file__).parent

    cfg = OmegaConf.load(file_path / "conf" / "config.yaml")

    df = load_wine()
    samples, df = extract_samples(df,
                                  samples=cfg.samples, 
                                  random_state=cfg.random_state)

    output_path = file_path / "data"
    df.to_csv(output_path / "ml.csv", index=False, sep=",")
    samples.to_csv(output_path / "samples.csv", index=False, sep=",")

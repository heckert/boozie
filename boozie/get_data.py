import io
import requests
import tempfile
import zipfile

from pathlib import Path
from typing import Tuple

import pandas as pd
from omegaconf import OmegaConf


def load_wine(transform_quality_to_stars: bool = True) -> pd.DataFrame:

    url = "https://archive.ics.uci.edu/static/public/186/wine+quality.zip"

    response = requests.get(url)

    if response.status_code == 200:

        with tempfile.TemporaryDirectory() as tmpdirname:

            zip_file = zipfile.ZipFile(io.BytesIO(response.content))

            zip_file.extractall(tmpdirname)
            zip_file.close()

            #red = pd.read_csv(
            #    Path(tmpdirname) / "winequality-red.csv", sep=";"
            #)
            white = pd.read_csv(
                Path(tmpdirname) / "winequality-white.csv", sep=";"
            )

        df = pd.concat([white], ignore_index=True)
        if transform_quality_to_stars:
            df = (df.assign(stars=lambda df: transform_to_star_score(df["quality"]))
                    .drop("quality", axis=1)
            )

    else:
        response.raise_for_status()

    return df


def transform_to_star_score(scores: pd.Series) -> pd.Series:
    result = pd.Series(index=scores.index)
    result[scores.isin(range(5))] = 1
    result[scores == 5] = 2
    result[scores == 6] = 3
    result[scores == 7] = 4
    result[scores > 7] = 5

    return result


def extract_samples(
    df: pd.DataFrame,*, score_name: str, samples: dict, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    df = df.copy()
    samples_df = pd.DataFrame()

    for name, score in samples.items():
        subset = df[df[score_name] == score]

        record = (subset.sample(random_state=random_state)
                        .assign(name=name))
        df = df.drop(record.index)
        samples_df = pd.concat([samples_df, record])

    return samples_df, df


def main() -> None:
    file_path = Path(__file__).parent

    cfg = OmegaConf.load(file_path / "conf" / "config.yaml")

    df = load_wine()
    samples, df = extract_samples(df,
                                  score_name=cfg.target,
                                  samples=cfg.samples, 
                                  random_state=cfg.random_state)

    output_path = file_path / "data"
    df.to_csv(output_path / "ml.csv", index=False, sep=",")
    samples.to_csv(output_path / "samples.csv", index=False, sep=",")


if __name__ == "__main__":
    main()

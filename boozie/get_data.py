import io
import requests
import tempfile
import zipfile

from pathlib import Path
from typing import Tuple

import pandas as pd

from imblearn.over_sampling import RandomOverSampler
from omegaconf import OmegaConf


def load_wine(transform_quality_to_stars: bool = True,
              run_oversampling: bool = True,
              random_state: int = 42) -> pd.DataFrame:

    TARGET = "quality"

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
        if transform_quality_to_stars:
            df = (df.assign(stars=lambda df: transform_to_star_score(df[TARGET]))
                    .drop(TARGET, axis=1)
            )
            TARGET = "stars"

        if run_oversampling:
            df = oversample(df, target=TARGET, random_state=random_state)

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


def oversample(df: pd.DataFrame, target: str, random_state: int = 42) -> pd.DataFrame:
    ros = RandomOverSampler(random_state=random_state)
    X = df.copy()
    y = X.pop(target)
    return pd.concat(ros.fit_resample(X, y), axis=1)


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

    df = load_wine(transform_quality_to_stars=cfg.transform_quality_to_stars,
                   run_oversampling=cfg.run_oversampling,
                   random_state=cfg.random_state)
    
    samples, df = extract_samples(df,
                                  score_name=cfg.target,
                                  samples=cfg.samples, 
                                  random_state=cfg.random_state)

    output_path = file_path / "data"
    df.to_csv(output_path / "ml.csv", index=False, sep=",")
    samples.to_csv(output_path / "samples.csv", index=False, sep=",")


if __name__ == "__main__":
    main()

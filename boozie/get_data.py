import io
import requests
import tempfile
import zipfile

from pathlib import Path

import pandas as pd


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

        df = pd.concat([red, white])

    else:
        response.raise_for_status()

    return df


if __name__ == "__main__":

    df = load_wine()
    output_path = Path(__file__).parent / "data" / "wine.csv"
    df.to_csv(output_path, index=False, sep=";")

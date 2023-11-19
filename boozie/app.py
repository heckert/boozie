from pathlib import Path

import dash
from dash import dcc, html
import pandas as pd

path = Path(__file__).parent

df = pd.read_csv(path / "data" / "ml.csv")

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Machine Learning Model Trainer"),
    dcc.Dropdown(
        id='column-selector',
        options=[{'label': col, 'value': col} for col in df.columns],
        multi=True,
        value=df.columns[:5].tolist()  # Default to first 5 columns
    ),
    html.Button('Train Model', id='train-button', n_clicks=0),
    dcc.Graph(id='accuracy-bar-plot')
])


if __name__ == '__main__':
    app.run_server(debug=False)

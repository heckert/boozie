from dataclasses import dataclass
from pathlib import Path

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd

from boozie.ml.model import ModelTrainer


path = Path(__file__).parent

df = pd.read_csv(path / "data" / "ml.csv")

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Machine Learning Model Trainer"),
    dcc.Dropdown(
        id='column-selector',
        options=[{'label': col, 'value': col} for col in df.columns[:-1]],
        multi=True,
        value=df.columns[:5].tolist()  # Default to first 5 columns
    ),
    html.Button('Train Model', id='train-button', n_clicks=0),
    dcc.Graph(id='accuracy-bar-plot')
])


@dataclass
class TrainingResult:
    name: str
    mse: float


def train_model(df):
    trainer = ModelTrainer(df, target="quality")
    trainer.train()

    return TrainingResult(trainer.name, trainer.evaluate())


results = []

@app.callback(
    Output('accuracy-bar-plot', 'figure'),
    [Input('train-button', 'n_clicks')],
    [State('column-selector', 'value')]
)
def update_bar_plot(n_clicks, selected_columns):

    if n_clicks > 0:
        results.append(train_model(df[selected_columns + ["quality"]]))
        data = pd.DataFrame(results)
        figure = px.bar(data, x='name', y='mse', title='Mean Squared Errors')
        return figure
    return {}


if __name__ == '__main__':
    app.run_server(debug=True)

from dataclasses import dataclass
from pathlib import Path

from typing import Optional

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from omegaconf import OmegaConf
from sklearn.pipeline import Pipeline

from ml.model import ModelTrainer


path = Path(__file__).parent
df = pd.read_csv(path / "data" / "ml.csv")

if 'results' not in st.session_state:
    st.session_state['results'] = []


# Streamlit app
def main():

    # Parse config
    file_path = Path(__file__).parent
    cfg = OmegaConf.load(file_path / "conf" / "config.yaml")

    st.title("Wine Quality Predictor")

    # Allow the user to select up to 5 features
    features = st.multiselect("Select features", list(df.columns[:-1]), default=list(df.columns)[:5])

    if len(features) > cfg.max_features:
        st.error(f"Please select no more than {cfg.max_features} features.")
        return

    # Button to train the model
    if st.button("Train Model"):
        # Train and evaluate the model
        train_and_evaluate_model(features, target=cfg.target)

    # Display results
    update_plot()

    # Reset button
    if st.button("Reset"):
        reset_results()

    # Selectbox for choosing a single model
    model_choices = [f'Run {i+1}' for i in range(len(st.session_state['results']))]
    selected_model: str = st.selectbox("Select a Model for Prediction", model_choices)
    
    # Load the samples dataset
    samples = (pd.read_csv(path / "data" / "samples.csv")
                 .sort_values(cfg.target, ascending=False)
    )

    # Button to make predictions
    if st.button("Make Predictions"):
        predictions = get_predictions(selected_model, samples)
        # TODO
        # * add predictions do dataframe
        # color columns 
        # add metrics
        # https://docs.streamlit.io/library/api-reference/data/st.metric
        

    st.dataframe(
        samples[["name", cfg.target]], 
        hide_index=True,
        column_config={
            "name": "Brand",
            cfg.target: st.column_config.NumberColumn(
                "Expert Rating",
                format="%d ⭐"
            )
        }
    )


@dataclass
class TrainingResult:
    mse: float
    model: Pipeline
    features: list


def train_and_evaluate_model(features: list, *, target: str):
    trainer = ModelTrainer(df[features], df[target])
    trainer.train()

    result = TrainingResult(trainer.evaluate(), trainer.model, features)

    st.session_state['results'].append(result)


def update_plot():

    df = pd.DataFrame(st.session_state['results'])

    plt.figure(figsize=(10, 6))

    plt.xlabel('Model Run')
    plt.ylabel('Mean Squared Error\n(lower is better)')
    plt.title('Model Performance Over Runs')

    if len(df) > 0:
        plt.bar(df.index, df.mse, color='blue')
        plt.xticks(df.index, [f'Run {i+1}' for i in df.index])
    else:
        plt.xticks([], [])

    # Display the plot in the Streamlit app
    st.pyplot(plt)


def get_predictions(selected_model: str, samples: pd.DataFrame) -> pd.Series:
    if not selected_model:
        st.error("Please select a model.")
        return

    run_number = int(selected_model.split(' ')[-1]) - 1

    training_result = st.session_state['results'][run_number]
    model = training_result.model
    features = training_result.features

    return pd.Series(
        model.predict(samples[features])
    )


def reset_results():
    st.session_state['results'] = []
    st.rerun()
    

if __name__ == "__main__":
    main()

from dataclasses import dataclass
from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.pipeline import Pipeline

from ml.model import ModelTrainer


path = Path(__file__).parent
df = pd.read_csv(path / "data" / "ml.csv")

if 'results' not in st.session_state:
    st.session_state['results'] = []


# Streamlit app
def main():
    st.title("Wine Quality Predictor")

    # Allow the user to select up to 5 features
    features = st.multiselect("Select features", list(df.columns[:-1]), default=list(df.columns)[:5])

    if len(features) > 5:
        st.error("Please select no more than 5 features.")
        return

    # Button to train the model
    if st.button("Train Model"):
        # Train and evaluate the model
        train_and_evaluate_model(features)

    # Display results
    update_plot()

    # Reset button
    if st.button("Reset"):
        reset_results()


@dataclass
class TrainingResult:
    mse: float
    model: Pipeline
    features: list


def train_and_evaluate_model(features):
    trainer = ModelTrainer(df[features], df["quality"])
    trainer.train()

    result = TrainingResult(trainer.evaluate(), trainer.model, features)

    st.session_state['results'].append(result)


def update_plot():

    df = pd.DataFrame(st.session_state['results'])

    plt.figure(figsize=(10, 6))

    plt.xlabel('Model Run')
    plt.ylabel('Mean Squared Error')
    plt.title('Model Performance Over Runs')

    if len(df) > 0:
        plt.bar(df.index, df.mse, color='blue')
        plt.xticks(df.index, [f'Run {i+1}' for i in df.index])
    else:
        plt.xticks([], [])

    # Display the plot in the Streamlit app
    st.pyplot(plt)


def reset_results():
    st.session_state['results'] = []
    st.rerun()
    

if __name__ == "__main__":
    main()

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

    # TODO: Put in cofig
    if len(features) > 12:
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

    # Selectbox for choosing a single model
    model_choices = [f'Run {i+1}' for i in range(len(st.session_state['results']))]
    selected_model = st.selectbox("Select a Model for Prediction", model_choices)

    # Load the separate dataset
    prediction_df = pd.read_csv(path / "data" / "samples.csv")

    # Button to make predictions
    if st.button("Make Predictions"):
        make_predictions(selected_model, prediction_df)

    # TODO
    # add rank, result summary, coloring


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
    plt.ylabel('Mean Squared Error\n(lower is better)')
    plt.title('Model Performance Over Runs')

    if len(df) > 0:
        plt.bar(df.index, df.mse, color='blue')
        plt.xticks(df.index, [f'Run {i+1}' for i in df.index])
    else:
        plt.xticks([], [])

    # Display the plot in the Streamlit app
    st.pyplot(plt)


def make_predictions(selected_model, prediction_df):
    if not selected_model:
        st.error("Please select a model.")
        return

    run_number = int(selected_model.split(' ')[-1]) - 1
    training_result = st.session_state['results'][run_number]
    model = training_result.model
    features = training_result.features

    # Generate predictions
    predictions = model.predict(prediction_df[features])

    # Add predictions to the DataFrame and sort
    prediction_df['predictions'] = predictions

    # TODO
    # Move to seperate function refine_display_df
    prediction_df = prediction_df[["name", "predictions", "quality"] + features]
    sorted_df = (prediction_df.sort_values(by='predictions', ascending=False)
                              .reset_index(drop=True)
                              .rename(columns={"quality": "actual quality score"}))

    # Display the sorted DataFrame
    st.write("Predictions Ordered by Model:", sorted_df)

                              

def reset_results():
    st.session_state['results'] = []
    st.rerun()
    

if __name__ == "__main__":
    main()

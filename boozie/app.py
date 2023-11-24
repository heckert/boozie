from dataclasses import dataclass
from pathlib import Path

from typing import Optional, Tuple

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from omegaconf import OmegaConf
from sklearn.pipeline import Pipeline

from ml.model import train_and_evaluate_model


# TODO:
# CHECK optimize model
# Optimize UI 
# * match column
# * metric x/5 correct
# * baloons
# Generate metadata
# Visualize metadata
# Add in features

# Set up module level variables
path = Path(__file__).parent
cfg = OmegaConf.load(path / "conf" / "config.yaml")

# Training dataframe
tdf = pd.read_csv(path / "data" / "ml.csv")
# Sample dataframe -> illustrated for evaluation
sdf = (
    pd.read_csv(path / "data" / "samples.csv")
      .sort_values(cfg.target, ascending=False)
)

sdf_keys = sdf[["name", cfg.target]]

# Set up session state
if "training_runs" not in st.session_state:
    st.session_state["training_runs"] = []


def main():

    st.title("🍷 Wine Quality Predictor")
    st.header("🧪 Step 1: Train a model")

    # Allow the user to select up to 5 features
    features = st.multiselect("Select features", 
                              cfg.features,
                              default=cfg.features[:cfg.max_features])

    if len(features) > cfg.max_features:
        st.error(f"Please select no more than {cfg.max_features} features.")
        return

    # Button to train the model
    if st.button("Train Model"):
        if len(features) == 0:
            st.error(f"You must select at least one column to train a model.")
            return
        if len(st.session_state["training_runs"]) > cfg.max_training_runs:
            st.error(f"You can create no more than {cfg.max_training_runs} models.")
            return

        # Train and evaluate the model
        result = train_and_evaluate_model(tdf[features], tdf[cfg.target])
        st.session_state["training_runs"].append(result)

    # Display results
    update_plot()

    # Reset button
    if st.button("Reset"):
        reset_training_runs()
    
    st.header("✨ Step 2: Predict the quality")

    # Selectbox for choosing a single model
    model_choices = [f"Model {i+1}" for i in range(len(st.session_state["training_runs"]))]
    selected_model: str = st.selectbox("Select a Model for Prediction", model_choices, index=0)


    column_formats = {
        "name": "Brand",
        cfg.target: st.column_config.NumberColumn("Expert Rating", format="%d ⭐")
    }

    # Button to make predictions
    if st.button("Make Predictions"):
        if len(st.session_state["training_runs"]) == 0:
            st.error(f"You must train at least one model before you can predict.")
            return

        model, features = get_model_and_features_from_session_state(selected_model)

        predictions = get_predictions(sdf[features], model)
        sdf_keys["predictions"] = predictions.round(cfg.decimals)

        decimal_formats = {1: "%.1f ⭐", 0: "%d ⭐"}
        column_formats["predictions"] = st.column_config.NumberColumn("Predicted Quality", format=decimal_formats[cfg.decimals])
        # TODO
        # * Match column



    st.dataframe(sdf_keys, hide_index=True, column_config=column_formats)


def update_plot():

    df = pd.DataFrame(st.session_state["training_runs"])

    plt.figure(figsize=(10, 6))

    plt.xlabel("Model Run")
    plt.ylabel("Mean Squared Error\n(lower is better)")
    plt.title("Model Performance Over Runs")

    if len(df) > 0:
        plt.bar(df.index, df.mse, color="blue")
        plt.xticks(df.index, [f"Model {i+1}" for i in df.index])
    else:
        plt.xticks([], [])

    # Display the plot in the Streamlit app
    st.pyplot(plt)


def get_model_and_features_from_session_state(model_name: str) -> Tuple[Pipeline, list]:
    run_number = int(model_name.split(" ")[-1]) - 1

    training_result = st.session_state["training_runs"][run_number]
    return training_result.model, training_result.features


def get_predictions(X: pd.DataFrame, model: Pipeline) -> pd.Series:
    return pd.Series(
        model.predict(X),
        index=X.index
    )


def reset_training_runs():
    st.session_state["training_runs"] = []
    st.rerun()
    

if __name__ == "__main__":
    main()

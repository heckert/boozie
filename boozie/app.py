import yaml

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from omegaconf import OmegaConf
from sklearn.pipeline import Pipeline

from boozie.ml.model import train_and_evaluate_model


# Set up module level variables
path = Path(__file__).parent
cfg = OmegaConf.load(path / "conf" / "config.yaml")

# Training dataframe
train_data = pd.read_csv(path / "data" / "ml.csv")
# Rating dataframe -> illustrated for evaluation
rating_data_full = pd.read_csv(path / "data" / "samples.csv").sort_values(
    cfg.target, ascending=False
)
rating_data_display = rating_data_full[["name", cfg.target]].copy()

metadata = OmegaConf.load(path / "data" / "metadata.yaml")
metadata = OmegaConf.to_container(metadata)
metadata = {key: metadata[key] for key in cfg.features}


# Set up session state
if "training_runs" not in st.session_state:
    st.session_state["training_runs"] = []


def main():
    # Attempting to solve autorefresh issue on GCP
    # https://discuss.streamlit.io/t/how-to-stop-manage-autorefresh/27970/3
    # Refresh hourly, hopefully to stop refreshes every few minutes.
    st_autorefresh(3.6e6, key="fizzbuzzcounter")

    st.title("🍷 Wine Quality Predictor")

    st.header("🔍 Step 0: Understand the data")

    # Create an expander for feature metadata
    with st.expander("View metadata"):
        if cfg.show_metadata:
            for feature, description in metadata.items():
                st.markdown(f"**{feature.title()}**  \n{description}")
        else:
            st.write("🤷🏻‍♂️ Sorry no metadata available for you.")


    st.header("🧪 Step 1: Train a model")

    # Allow the user to select up to x features
    with st.container():
        features = st.multiselect(
            f"Select up to {cfg.max_features} features", cfg.features, max_selections=cfg.max_features
        )

    if len(st.session_state["training_runs"]) < cfg.max_training_runs:
        #if len(features) <= cfg.max_features:
        if st.button("Train Model"):
            if len(features) == 0:
                st.error("You must select at least one column to train a model.")
                return

            # Train and evaluate the model
            result = train_and_evaluate_model(train_data[features], train_data[cfg.target])
            st.session_state["training_runs"].append(result)

        #else:
        #    st.error(f"Please select no more than {cfg.max_features} features.")

    else:
        st.info(f"You can create no more than {cfg.max_training_runs} models. Press reset to continue training.")

    # Display results
    update_plot()

    # Reset button
    if st.button("Reset"):
        reset_training_runs()

    st.header("✨ Step 2: Predict the quality")

    # Selectbox for choosing a single model
    model_choices = [
        f"Model {i+1}" for i in range(len(st.session_state["training_runs"]))
    ]
    selected_model: str = st.selectbox(
        "Select a Model for Prediction", model_choices, index=0
    )

    column_formats = {
        "name": "Brand",
        cfg.target: st.column_config.NumberColumn("Expert Rating", format="%d ⭐"),
    }
    
    # Variable for displaying result metric
    n_matches = None

    # Button to make predictions
    if st.button("Make Predictions"):
        if len(st.session_state["training_runs"]) == 0:
            st.error("You must train at least one model before you can predict.")
            return

        model, features = get_model_and_features_from_session_state(selected_model)

        predictions = get_predictions(rating_data_full[features], model)
        rating_data_display["predictions"] = predictions.round(cfg.decimals)

        decimal_formats = {1: "%.1f ⭐", 0: "%d ⭐"}
        column_formats["predictions"] = st.column_config.NumberColumn(
            "Predicted Quality", format=decimal_formats[cfg.decimals]
        )

        # Assign match columns
        matcher = RoundMatcher()
        rating_data_display["match"] = matcher.get_matches(
            rating_data_display[cfg.target], rating_data_display["predictions"]
        )
        column_formats["match"] = "Match"

        n_matches = matcher.n_matches


    col1, col2 = st.columns(2)

    with col1:
        st.dataframe(rating_data_display, hide_index=True, column_config=column_formats)

    if n_matches is not None:
        col2.metric("Correct results", value=f"{n_matches}/5")

        if n_matches == 5:
            st.balloons()


def update_plot():
    df = pd.DataFrame(st.session_state["training_runs"])

    plt.figure(figsize=(10, 3))

    plt.xlabel("Training Run")
    plt.ylabel("Mean Squared Error")
    plt.title("Training Performance [less is better]")

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
    return pd.Series(model.predict(X), index=X.index)


class RoundMatcher:
    n_matches: int

    def get_matches(
        self, integers: pd.Series, floats: pd.Series, decimals=0
    ) -> pd.Series:
        
        labels = {True: "✅", False: "❌"}
        
        bools = integers == floats.round(decimals)

        self.n_matches = bools.sum()

        return bools.replace(labels)


def reset_training_runs():
    st.session_state["training_runs"] = []
    st.rerun()


if __name__ == "__main__":
    main()

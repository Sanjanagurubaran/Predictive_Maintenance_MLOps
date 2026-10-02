import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostClassifier

# Page configuration
st.set_page_config(
    page_title="Predictive Maintenance System",
    page_icon="⚙️",
    layout="wide"
)

# Model location
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "predictive_maintenance_catboost.cbm"

# Exact features used during CatBoost training
FEATURE_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

# Load the saved model once
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    loaded_model = CatBoostClassifier()
    loaded_model.load_model(str(MODEL_PATH))
    return loaded_model


# Load model and display errors clearly
try:
    model = load_model()
except Exception as error:
    st.error(f"Unable to load the CatBoost model: {error}")
    st.stop()


# Application heading
st.title("⚙️ Predictive Maintenance System")
st.subheader(
    "Industrial Equipment Failure Prediction Using CatBoost and MLOps"
)
st.write(
    "Enter the equipment operating parameters to predict "
    "whether the machine condition is Normal or Failure."
)

st.divider()

# Equipment input form
st.header("🔧 Equipment Parameters")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        air_temp = st.number_input(
            "Air Temperature [K]",
            min_value=250.0,
            max_value=350.0,
            value=300.0,
            step=0.1
        )

        rotational_speed = st.number_input(
            "Rotational Speed [rpm]",
            min_value=0,
            max_value=5000,
            value=1500,
            step=10
        )

        tool_wear = st.number_input(
            "Tool Wear [min]",
            min_value=0,
            max_value=500,
            value=100,
            step=1
        )

    with col2:
        process_temp = st.number_input(
            "Process Temperature [K]",
            min_value=250.0,
            max_value=400.0,
            value=310.0,
            step=0.1
        )

        torque = st.number_input(
            "Torque [Nm]",
            min_value=0.0,
            max_value=200.0,
            value=40.0,
            step=0.1
        )

    submitted = st.form_submit_button(
        "🔍 Predict Equipment Condition",
        use_container_width=True
    )


# Generate prediction
if submitted:
    input_data = pd.DataFrame(
        [[
            air_temp,
            process_temp,
            rotational_speed,
            torque,
            tool_wear
        ]],
        columns=FEATURE_COLUMNS
    )

    try:
        prediction = int(
            np.asarray(model.predict(input_data)).reshape(-1)[0]
        )

        probabilities = model.predict_proba(input_data)[0]
        failure_probability = float(probabilities[1])

        st.divider()
        st.header("📊 Prediction Result")

        result_col, probability_col = st.columns(2)

        with result_col:
            if prediction == 1:
                st.error("⚠️ Predicted Condition: FAILURE")
                st.warning(
                    "The model predicts a potential failure condition. "
                    "Inspect the equipment and validate the result "
                    "with maintenance personnel."
                )
            else:
                st.success("✅ Predicted Condition: NORMAL")
                st.info(
                    "The model predicts a normal condition for these "
                    "input values. Continue routine monitoring."
                )

        with probability_col:
            st.metric(
                "Model-estimated failure probability",
                f"{failure_probability * 100:.2f}%"
            )
            st.progress(failure_probability)

        st.subheader("Entered Equipment Data")
        st.dataframe(input_data, use_container_width=True)

    except Exception as error:
        st.error(f"Prediction failed: {error}")


st.divider()
st.caption(
    "CatBoost-based predictive maintenance demonstration. "
    "Predictions are based on the trained dataset model and "
    "should be verified before maintenance decisions."
)
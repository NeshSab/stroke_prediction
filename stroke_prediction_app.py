"""
Stroke Risk Prediction App
This application allows users to predict stroke risk based on demographic and medical
inputs.
- The "Stroke Risk Prediction" tab allows users to input relevant health data and
    get predictions.
- The "About" tab provides an overview of the project.
"""

import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt


tab1, tab2 = st.tabs(["🩺 Stroke Risk Prediction", "About"])


@st.cache_resource
def load_model():
    """Loads the pre-trained LightGBM model."""
    return joblib.load("lightgbm.joblib")


model = load_model()

with tab1:
    st.title("Stroke Risk Prediction")
    st.write("Enter the details below to predict stroke risk.")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Select", "Male", "Female", "Other"])
        age = st.number_input("Age", min_value=1, max_value=120, value=None)
        bmi = st.number_input(
            "BMI (Optional)", min_value=10.0, max_value=100.0, value=None, step=0.1
        )
        avg_glucose_level = st.number_input(
            "Avg Glucose Level", min_value=50.0, max_value=300.0, value=None
        )
        ever_married = st.selectbox("Ever Married", ["Select", "No", "Yes"])

    with col2:
        hypertension = st.selectbox("Hypertension", ["Select", "No", "Yes"])
        heart_disease = st.selectbox("Heart Disease", ["Select", "No", "Yes"])
        smoking_status = st.selectbox(
            "Smoking Status",
            ["Select", "Former Smoker", "Never Smoked", "Smoker", "Unknown"],
        )

        urban_resident = st.selectbox("Residence Type", ["Select", "Urban", "Rural"])

        if age and age > 23:
            work_type_options = ["Select", "Private", "Self Employed", "Government"]
        else:
            work_type_options = [
                "Select",
                "Private",
                "Self Employed",
                "Government",
                "Unemployed",
            ]
        work_type = st.selectbox("Work Type", work_type_options)

    if st.button("Predict"):
        if (
            "Select"
            in [
                gender,
                hypertension,
                heart_disease,
                ever_married,
                work_type,
                urban_resident,
                smoking_status,
            ]
            or age is None
            or avg_glucose_level is None
        ):
            st.error("⚠️ Please fill in all required fields before predicting.")
        else:
            binary_mapping = {"No": 0, "Yes": 1}
            hypertension = binary_mapping[hypertension]
            heart_disease = binary_mapping[heart_disease]
            ever_married = binary_mapping[ever_married]

            residenc_type_mapping = {"Urban": 1, "Rural": 0}
            urban_resident = residenc_type_mapping[urban_resident]

            gender_mapping = {"Male": 0, "Female": 1, "Other": 2}
            gender = gender_mapping[gender]

            work_type_mapping = {
                "Private": 0,
                "Self Employed": 1,
                "Government": 2,
                "Unemployed": 3,
            }
            work_type = work_type_mapping[work_type]

            smoking_status_mapping = {
                "Former Smoker": 0,
                "Never Smoked": 1,
                "Smoker": 2,
                "Unknown": 3,
            }
            smoking_status = smoking_status_mapping[smoking_status]

            bmi_missing = 1 if bmi is None else 0
            bmi = 28 if bmi is None else bmi

            feature_names = [
                "gender",
                "age",
                "hypertension",
                "heart_disease",
                "ever_married",
                "work_type",
                "urban_resident",
                "avg_glucose_level",
                "bmi",
                "smoking_status",
                "bmi_missing",
            ]

            input_data_df = pd.DataFrame(
                [
                    [
                        gender,
                        age,
                        hypertension,
                        heart_disease,
                        ever_married,
                        work_type,
                        urban_resident,
                        avg_glucose_level,
                        bmi,
                        smoking_status,
                        bmi_missing,
                    ]
                ],
                columns=feature_names,
            )

            preprocessor = model.named_steps["preprocessor"]
            model_only = model.named_steps["model"]
            transformed_data = preprocessor.transform(input_data_df)
            transformed_data = pd.DataFrame(transformed_data, columns=feature_names)

            prediction = model_only.predict(transformed_data)[0]
            result = "🛑 High Stroke Risk" if prediction == 1 else "✅ Low Stroke Risk"
            st.subheader(f"Prediction: {result}")

            st.subheader("SHAP Explanation")
            explainer = shap.TreeExplainer(model_only)
            shap_values = explainer.shap_values(
                transformed_data, check_additivity=False
            )

            plt.style.use("dark_background")
            fig, ax = plt.subplots(figsize=(8, 4))
            shap.summary_plot(shap_values, transformed_data, show=False)
            plt.xlabel("SHAP Value", color="white")
            plt.ylabel("Feature", color="white")
            plt.xticks(color="white")
            plt.yticks(color="white")
            st.pyplot(fig)

with tab2:
    st.title("About the Stroke Prediction App")
    st.write(
        """
    ### 📌 Overview
    This application predicts the **risk of stroke** based on patient demographics, medical history, and lifestyle factors.  
    - Uses LightGBM model trained on health data.**
    - Developed as part of a **data science project** on stroke risk prediction.

    ### ⚙️ How It Works
    1️⃣ **Enter patient details** (e.g., age, gender, medical history, lifestyle).  
    2️⃣ **Model processes inputs** and predicts whether the patient is at **high or low risk** for stroke.  
    3️⃣ **SHAP values** explain which features influenced the prediction the most.

    ### 📊 Data Used
    - The dataset contains **5,110 patient records** with features like `age`, `hypertension`, `bmi`, `glucose level`, and `smoking status`.  
    - Dataset source: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/data
    """
    )

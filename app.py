import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# 1. Page Configuration
st.set_page_config(
    page_title="Heart Disease Multi-Model Dashboard",
    page_icon="❤️",
    layout="wide"  # wide layout: Bteer for Dashboard
)

# Prevent text wrapping in the sidebar by setting a minimum width
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            min-width: 380px;
            max-width: 500px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# 2. # Load all models at the start of the ap
@st.cache_resource      # Cache Model Loading
def load_all_models():
    models_dir = Path("models")
    model_files = {
        "Logistic Regression (Best Model)": models_dir / "best_logistic_regression_model.pkl",
        "Random Forest": models_dir / "best_random_forest_model.pkl",
        "SVM": models_dir / "best_svm_model.pkl",
        "KNN": models_dir / "best_knn_model.pkl"
    }
    
    loaded_models = {}
    for name, path in model_files.items():
        if path.exists():
            loaded_models[name] = joblib.load(path)
    return loaded_models

models = load_all_models()

# ----------------- Sidebar: Dashboard Info -----------------
st.sidebar.title("Dashboard Settings")
mode = st.sidebar.radio("View Mode", ["Multi-Model Comparison", "Single Model Deep-Dive"])

st.sidebar.markdown("---")
st.sidebar.subheader("Model Performance (Test Set)")
st.sidebar.markdown("""
* **Logistic Regression**: Acc `88.52%` | AUC `0.92`
* **Random Forest**: Acc `86.89%` | AUC `0.94`
* **SVM**: Acc `85.25%` | AUC `0.93`
* **KNN**: Acc `83.61%` | AUC `0.93`
""")

# ----------------- Main Page Content -----------------
st.title("Heart Disease Clinical Prediction Dashboard")
st.write("Enter the patient's clinical parameters below to generate and compare predictions across 4 tuned Machine Learning models.")

# Input Form
with st.form("prediction_form"):
    st.subheader("Patient Clinical Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=54)
        sex = st.selectbox("Sex (1 = Male, 0 = Female)", [1, 0])
        cp = st.selectbox("Chest Pain Type (cp: 0-3)", [0, 1, 2, 3])
        trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=80, max_value=200, value=120)
        
    with col2:
        chol = st.number_input("Serum Cholesterol (chol in mg/dl)", min_value=100, max_value=600, value=240)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
        restecg = st.selectbox("Resting ECG Results (restecg: 0-2)", [0, 1, 2])
        thalach = st.number_input("Max Heart Rate Achieved (thalach)", min_value=60, max_value=220, value=150)
        
    with col3:
        exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])
        oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        slope = st.selectbox("Slope of ST Segment (slope: 0-2)", [0, 1, 2])
        ca = st.selectbox("Major Vessels Colored by Flourosopy (ca: 0-3)", [0, 1, 2, 3])
        thal = st.selectbox("Thal (thal: 1 = Normal, 2 = Fixed Defect, 3 = Reversable Defect)", [1, 2, 3])

    submit_button = st.form_submit_button(label="Generate Predictions", use_container_width=True)

# ----------------- Prediction Results -----------------
if submit_button:
    if not models:
        st.error("No saved models found in `models/` directory! Please make sure your `.pkl` files are placed correctly.")
    else:
        # 1. Create Feature DataFrame
        input_data = pd.DataFrame([[
            age, sex, cp, trestbps, chol, fbs, restecg, 
            thalach, exang, oldpeak, slope, ca, thal
        ]], columns=[
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ])

        st.markdown("---")
        
        # ----------------- Mode 1: Multi-Model Comparison -----------------
        if mode == "Multi-Model Comparison":
            st.subheader("4-Model Comparative Analysis")
            
            # 4 Column Layout for Model Predictions
            cols = st.columns(len(models))
            
            for idx, (model_name, model_obj) in enumerate(models.items()):
                with cols[idx]:
                    st.markdown(f"#### {model_name}")
                    pred = model_obj.predict(input_data)[0]
                    
                    # Probability
                    proba = model_obj.predict_proba(input_data)[0] if hasattr(model_obj, "predict_proba") else None
                    
                    if pred == 1:
                        st.error("**High Risk**")
                        if proba is not None:
                            disease_prob = proba[1] * 100
                            st.metric("Risk Probability", f"{disease_prob:.1f}%")
                            st.progress(int(disease_prob))
                    else:
                        st.success("**Low Risk**")
                        if proba is not None:
                            healthy_prob = proba[0] * 100
                            st.metric("Healthy Confidence", f"{healthy_prob:.1f}%")
                            st.progress(int(healthy_prob))

            # Interpretation of Model Consensus
            predictions = [m.predict(input_data)[0] for m in models.values()]
            high_risk_count = sum(predictions)
            
            st.markdown("---")
            st.subheader("Model Consensus Summary")
            if high_risk_count >= 3:
                st.warning(f"**High Clinical Concern**: {high_risk_count} out of {len(models)} models predict a **higher likelihood of Heart Disease**.")
            elif high_risk_count <= 1:
                st.info(f"**Low Clinical Concern**: {len(models) - high_risk_count} out of {len(models)} models predict **lower risk**.")
            else:
                st.write(f"**Borderline Case**: Models are split ({high_risk_count} High Risk vs {len(models) - high_risk_count} Low Risk). Logistic Regression (Champion Model) is recommended as primary reference.")

        # ----------------- Mode 2: Single Model Deep-Dive -----------------
        else:
            selected_model_name = st.selectbox("Select Model to Inspect:", list(models.keys()))
            selected_model = models[selected_model_name]
            
            pred = selected_model.predict(input_data)[0]
            proba = selected_model.predict_proba(input_data)[0] if hasattr(selected_model, "predict_proba") else None
            
            st.subheader(f"Detailed Result: {selected_model_name}")
            
            res_col1, res_col2 = st.columns([1, 2])
            with res_col1:
                if pred == 1:
                    st.error("**Predicted Status: Heart Disease Detected**")
                else:
                    st.success("**Predicted Status: Normal / Low Risk**")
                    
            with res_col2:
                if proba is not None:
                    st.write(f"Probability of No Disease: `{proba[0]*100:.2f}%`")
                    st.write(f"Probability of Heart Disease: `{proba[1]*100:.2f}%`")
                    st.progress(int(proba[1]*100))
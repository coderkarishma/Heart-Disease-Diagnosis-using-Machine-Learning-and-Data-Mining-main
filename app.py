```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

.main {
    background-color: #f8fafc;
}

[data-testid="stSidebar"] {
    background: #0f172a;
}

[data-testid="stSidebar"] * {
    color: white;
}

.hero {
    padding: 35px;
    border-radius: 20px;
    background: linear-gradient(90deg,#ef4444,#f97316);
    color: white;
    text-align: center;
    margin-bottom: 20px;
}

.footer {
    text-align:center;
    color:gray;
    padding:20px;
}

.stButton > button {
    width:100%;
    border-radius:12px;
    height:55px;
    font-size:18px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# ---------------- MODEL ----------------

@st.cache_resource
def load_model():

    dataset = pd.read_csv(
        "Heart-Disease-Diagnosis-using-Machine-Learning-and-Data-Mining-main/dataset/heart_statlog_cleveland_hungary_final.csv"
    )

    mode_st_slope = dataset['ST slope'].mode()[0]
    dataset['ST slope'] = dataset['ST slope'].replace(0, mode_st_slope)

    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=0
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    model = RandomForestClassifier(
        n_estimators=20,
        criterion="entropy",
        random_state=0
    )

    model.fit(X_train, y_train)

    return model, scaler

model, scaler = load_model()

# ---------------- HEADER ----------------

st.markdown("""
<div class="hero">
    <h1>❤️ Heart Disease Prediction System</h1>
    <p>AI Powered Healthcare Risk Assessment Dashboard</p>
</div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------

st.sidebar.header("🩺 Patient Information")

age = st.sidebar.number_input("Age", 1, 120, 45)

sex = st.sidebar.selectbox(
    "Gender",
    [0, 1],
    format_func=lambda x: "Female" if x == 0 else "Male"
)

cp = st.sidebar.selectbox("Chest Pain Type", [1, 2, 3, 4])

trestbps = st.sidebar.number_input(
    "Resting Blood Pressure",
    value=120
)

chol = st.sidebar.number_input(
    "Cholesterol",
    value=200
)

fbs = st.sidebar.selectbox(
    "Fasting Blood Sugar",
    [0, 1]
)

restecg = st.sidebar.selectbox(
    "Resting ECG",
    [0, 1, 2]
)

thalach = st.sidebar.number_input(
    "Maximum Heart Rate",
    value=150
)

exang = st.sidebar.selectbox(
    "Exercise Induced Angina",
    [0, 1]
)

oldpeak = st.sidebar.number_input(
    "Oldpeak",
    value=1.0,
    step=0.1
)

slope = st.sidebar.selectbox(
    "ST Slope",
    [1, 2, 3]
)

# ---------------- PREDICTION ----------------

if st.button("🔍 Predict Heart Disease Risk"):

    features = np.array([[
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope
    ]])

    features = scaler.transform(features)

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    # Metrics

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Age", age)

    with col2:
        st.metric("Cholesterol", chol)

    with col3:
        st.metric(
            "Risk Score",
            f"{probability*100:.1f}%"
        )

    # Gauge Chart

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={'text': "Heart Disease Risk (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 30], 'color': 'lightgreen'},
                {'range': [30, 70], 'color': 'gold'},
                {'range': [70, 100], 'color': 'salmon'}
            ]
        }
    ))

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Risk Level

    if probability < 0.30:
        st.success(
            f"🟢 Low Risk ({probability*100:.2f}%)"
        )
    elif probability < 0.70:
        st.warning(
            f"🟡 Moderate Risk ({probability*100:.2f}%)"
        )
    else:
        st.error(
            f"🔴 High Risk ({probability*100:.2f}%)"
        )

    # Final Prediction

    if prediction == 1:
        st.error(
            "⚠️ Patient is likely to have Heart Disease"
        )
    else:
        st.success(
            "✅ Patient is Normal"
        )

    # Patient Summary

    st.info(f"""
### 📋 Patient Summary

- Age: {age}
- Gender: {"Male" if sex == 1 else "Female"}
- Cholesterol: {chol}
- Risk Score: {probability*100:.2f}%
""")

    # Health Tips

    if probability > 0.70:
        st.markdown("""
### 🩺 Recommended Actions

✅ Consult a Cardiologist

✅ Reduce Cholesterol Intake

✅ Exercise Regularly

✅ Monitor Blood Pressure

✅ Avoid Smoking

✅ Maintain Healthy Weight
""")

    # Download Report

    report = f"""
HEART DISEASE REPORT

Age: {age}
Gender: {"Male" if sex == 1 else "Female"}

Cholesterol: {chol}

Risk Score: {probability*100:.2f}%
"""

    st.download_button(
        "📄 Download Report",
        report,
        file_name="heart_disease_report.txt"
    )

# ---------------- FOOTER ----------------

st.markdown("""
<div class="footer">
❤️ Made with Streamlit & Machine Learning
</div>
""", unsafe_allow_html=True)
```

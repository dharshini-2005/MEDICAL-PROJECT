import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import plotly.express as px

# 🔐 Safe speech recognition import
try:
    import speech_recognition as sr
    VOICE_ENABLED = True
except:
    VOICE_ENABLED = False

# 🌐 Page Setup
st.set_page_config(
    page_title="AI Medical Diagnosis Dashboard",
    page_icon="🩺",
    layout="wide"
)

# 🎨 Custom Styling
st.markdown("""
<style>
.main { background-color: #f7f9fc; }
h1, h2, h3 { color: #0056b3 !important; }
.stButton>button {
    color: white;
    background-color: #0078D7;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-weight: 600;
    transition: 0.3s;
}
.stButton>button:hover {
    background-color: #005bb5;
    transform: scale(1.05);
}
.stTextInput>div>div>input {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# 🩺 Title
st.title("🩺 AI-Powered Medical Diagnosis Dashboard")
st.markdown("### Combining **Rule-based Logic** and **Machine Learning** for Symptom Analysis")
st.write("Provide your symptoms to predict possible diseases with interactive visualizations.")
st.divider()

# 📂 Load Dataset
try:
    df = pd.read_csv("rules_dataset/dataset.csv")
    desc_df = pd.read_csv("rules_dataset/symptom_Description.csv")
    precaution_df = pd.read_csv("rules_dataset/symptom_precaution.csv")
except FileNotFoundError:
    st.error("⚠️ Dataset files missing. Please check the rules_dataset folder.")
    st.stop()

# 🧹 Clean Data
df.columns = df.columns.str.strip()
desc_df.columns = desc_df.columns.str.strip()
precaution_df.columns = precaution_df.columns.str.strip()

desc_dict = dict(zip(desc_df["Disease"].str.lower(), desc_df["Description"]))

precaution_dict = {}
for _, row in precaution_df.iterrows():
    disease = row["Disease"].lower()
    precaution_dict[disease] = [
        str(row[col]) for col in precaution_df.columns
        if "Precaution" in col and pd.notna(row[col])
    ]

symptom_columns = [c for c in df.columns if "symptom" in c.lower()]

# 🧠 Rule-Based Knowledge
rules = []
for _, row in df.iterrows():
    symptoms = [str(row[c]).lower() for c in symptom_columns if pd.notna(row[c])]
    if symptoms:
        rules.append({
            "disease": row["Disease"],
            "symptoms": symptoms,
            "description": desc_dict.get(row["Disease"].lower(), "No description available."),
            "precautions": precaution_dict.get(row["Disease"].lower(), ["Consult a doctor"])
        })

rules = list({r["disease"]: r for r in rules}.values())
st.success(f"✅ Loaded {len(rules)} disease rules")

# ⚙️ Train ML Model
X = df[symptom_columns].notna().astype(int)
y = df["Disease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=120, random_state=42)
model.fit(X_train, y_train)

# 🎙️ Speech Function
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙️ Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        return recognizer.recognize_google(audio)

# 🩸 Input Symptoms
st.subheader("🩸 Input Your Symptoms")
col1, col2 = st.columns([2, 1])

with col1:
    user_input = st.text_input(
        "Enter symptoms (comma-separated)",
        placeholder="headache, fever, fatigue"
    )

with col2:
    if VOICE_ENABLED:
        if st.button("🎤 Speak Symptoms"):
            try:
                user_input = recognize_speech()
                st.success(f"You said: {user_input}")
            except:
                st.error("Voice input failed")
    else:
        st.info("🎤 Voice input disabled on cloud")

user_symptoms = [s.strip().lower() for s in user_input.split(",") if s.strip()]

# 🔍 Diagnose
if st.button("🔍 Diagnose Now"):
    if not user_symptoms:
        st.warning("⚠️ Please enter at least one symptom")
    else:
        st.subheader("🧩 Rule-Based Diagnosis")

        results = {}
        for rule in rules:
            common = set(user_symptoms) & set(rule["symptoms"])
            if common:
                score = len(common) / len(rule["symptoms"])
                if rule["disease"] not in results or score > results[rule["disease"]]["score"]:
                    results[rule["disease"]] = {
                        "score": score,
                        "info": rule,
                        "matched": list(common)
                    }

        if results:
            top = sorted(results.items(), key=lambda x: x[1]["score"], reverse=True)[:5]

            df_chart = pd.DataFrame({
                "Disease": [d for d, _ in top],
                "Match (%)": [v["score"] * 100 for _, v in top]
            })

            fig = px.bar(
                df_chart,
                x="Disease",
                y="Match (%)",
                color="Disease",
                text="Match (%)",
                title="🧠 Animated Disease Match Score",
                color_discrete_sequence=px.colors.sequential.Viridis
            )

            fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            fig.update_layout(transition_duration=800)
            st.plotly_chart(fig, use_container_width=True)

            for disease, data in top:
                with st.expander(f"🩺 {disease} ({data['score']*100:.0f}%)"):
                    st.write("**Description:**", data["info"]["description"])
                    st.write("**Matched Symptoms:**", ", ".join(data["matched"]))
                    st.write("**Precautions:**")
                    for p in data["info"]["precautions"]:
                        st.write("-", p)

        # 🤖 ML Prediction
        st.subheader("🤖 Machine Learning Prediction")

        input_vec = np.array([
            1 if s in user_symptoms else 0 for s in symptom_columns
        ]).reshape(1, -1)

        probs = model.predict_proba(input_vec)[0]
        top3 = np.argsort(probs)[::-1][:3]

        df_ml = pd.DataFrame({
            "Disease": model.classes_[top3],
            "Confidence (%)": probs[top3] * 100
        })

        fig_ml = px.bar(
            df_ml,
            x="Disease",
            y="Confidence (%)",
            color="Disease",
            text="Confidence (%)",
            title="🤖 ML Model Confidence",
            color_discrete_sequence=px.colors.sequential.Plasma
        )

        fig_ml.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig_ml.update_layout(transition_duration=900)
        st.plotly_chart(fig_ml, use_container_width=True)

st.divider()
st.caption("💡 Deployed on Streamlit Cloud | AI Medical Diagnosis System")

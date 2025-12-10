import pandas as pd
import streamlit as st
import speech_recognition as sr
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import plotly.express as px
import time

st.set_page_config(page_title="AI Medical Diagnosis Dashboard", page_icon=" ", layout="wide")

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
        .stTextInput>div>div>input { border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

st.title(" AI-Powered Medical Diagnosis Dashboard")
st.markdown("### Combining **Rule-based Logic** and **Machine Learning** for Symptom Analysis")
st.write("Provide or speak your symptoms to predict possible diseases with interactive visualizations.")
st.divider()


try:
    df = pd.read_csv("rules_dataset/dataset.csv")
    desc_df = pd.read_csv("rules_dataset/symptom_Description.csv")
    precaution_df = pd.read_csv("rules_dataset/symptom_precaution.csv")
except FileNotFoundError:
    st.error(" Missing CSV files! Ensure all 3 datasets are inside the `rules_dataset` folder.")
    st.stop()

df.columns = df.columns.str.strip()
desc_df.columns = desc_df.columns.str.strip()
precaution_df.columns = precaution_df.columns.str.strip()

desc_dict = dict(zip(desc_df["Disease"].str.strip().str.lower(), desc_df["Description"]))

precaution_dict = {}
for _, row in precaution_df.iterrows():
    disease = str(row["Disease"]).strip().lower()
    precautions = [str(row[col]).strip() for col in precaution_df.columns if "Precaution" in col and pd.notna(row[col])]
    precaution_dict[disease] = precautions

symptom_columns = [col for col in df.columns if "Symptom" in col or "symptom" in col]

rules = []
for _, row in df.iterrows():
    symptoms = [str(row[col]).strip().lower() for col in symptom_columns if pd.notna(row[col])]
    if symptoms:
        disease_name = str(row["Disease"]).strip()
        disease_key = disease_name.lower()
        rules.append({
            "symptoms": symptoms,
            "disease": disease_name,
            "description": desc_dict.get(disease_key, "No description available."),
            "precautions": precaution_dict.get(disease_key, ["Consult a doctor for proper guidance."])
        })
rules = list({rule["disease"]: rule for rule in rules}.values())

st.success(f"Loaded {len(rules)} unique disease rules from dataset.")

X = df[symptom_columns].notna().astype(int)
y = df["Disease"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=120, random_state=42)
model.fit(X_train, y_train)

def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙️ Listening... Speak your symptoms clearly.")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            st.success(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            st.error("Could not understand your voice.")
        except sr.RequestError:
            st.error(" Internet connection issue.")
        return ""

st.subheader(" Input Your Symptoms")
col1, col2 = st.columns([2, 1])
with col1:
    user_input = st.text_input("Enter your symptoms (comma-separated):", placeholder="e.g., headache, fever, fatigue")
with col2:
    if st.button(" Speak Symptoms"):
        spoken_text = recognize_speech()
        if spoken_text:
            user_input = spoken_text

user_symptoms = [s.strip().lower() for s in user_input.split(',') if s.strip()]

if st.button("Diagnose Now"):
    if not user_symptoms:
        st.warning("⚠️ Please enter at least one symptom.")
    else:
        st.subheader(" Rule-Based Diagnosis Results")

        results = {}
        for rule in rules:
            common = set(user_symptoms) & set(rule["symptoms"])
            if common:
                match_score = len(common) / len(rule["symptoms"])
                disease = rule["disease"]
                if disease not in results or match_score > results[disease]["score"]:
                    results[disease] = {
                        "score": match_score,
                        "description": rule["description"],
                        "precautions": rule["precautions"],
                        "matched_symptoms": list(common)
                    }

        if results:
            sorted_results = sorted(results.items(), key=lambda x: x[1]["score"], reverse=True)
            top_results = sorted_results[:5]

            chart_data = pd.DataFrame({
                "Disease": [d for d, _ in top_results],
                "Match Score (%)": [info["score"]*100 for _, info in top_results]
            })

            fig_match = px.bar(
                chart_data,
                x="Disease",
                y="Match Score (%)",
                color="Disease",
                color_discrete_sequence=px.colors.sequential.Viridis,
                title=" Animated Disease Match Score",
                text="Match Score (%)",
                animation_frame=None
            )
            fig_match.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            fig_match.update_layout(
                transition_duration=800,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis={'categoryorder': 'total descending'},
                yaxis_title="Match Confidence (%)",
                title_font_size=22
            )
            st.plotly_chart(fig_match, use_container_width=True)

            for disease, info in top_results:
                with st.expander(f" {disease} ({info['score']*100:.0f}% match)"):
                    st.markdown(f"**🧾 Description:** {info['description']}")
                    st.markdown(f"** Matched Symptoms:** {', '.join(info['matched_symptoms'])}")
                    st.markdown("** Precautions:**")
                    for p in info["precautions"]:
                        st.write(f"- {p}")
        else:
            st.warning("❗No close rule-based matches found.")

        st.subheader(" Machine Learning Predictions")

        input_vector = np.array([1 if sym in user_symptoms else 0 for sym in symptom_columns]).reshape(1, -1)
        ml_prob = model.predict_proba(input_vector)[0]
        top3_idx = np.argsort(ml_prob)[::-1][:3]
        top3_diseases = [(model.classes_[i], ml_prob[i]) for i in top3_idx]

        chart_ml = pd.DataFrame({
            "Disease": [d for d, _ in top3_diseases],
            "Confidence (%)": [p*100 for _, p in top3_diseases]
        })

        fig_ml = px.bar(
            chart_ml,
            x="Disease",
            y="Confidence (%)",
            color="Disease",
            color_discrete_sequence=px.colors.sequential.Plasma,
            text="Confidence (%)",
            title="ML Model Confidence per Disease"
        )
        fig_ml.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig_ml.update_layout(
            transition_duration=900,
            plot_bgcolor='rgba(255,255,255,0)',
            xaxis={'categoryorder': 'total descending'},
            title_font_size=22
        )
        st.plotly_chart(fig_ml, use_container_width=True)

        for disease, prob in top3_diseases:
            with st.expander(f" {disease} — {prob*100:.2f}% confidence"):
                disease_key = disease.lower()
                st.markdown(f"** Description:** {desc_dict.get(disease_key, 'No description available.')}") 
                st.markdown("** Precautions:**")
                for p in precaution_dict.get(disease_key, ['Consult a doctor for proper guidance.']):
                    st.write(f"- {p}")

st.divider()


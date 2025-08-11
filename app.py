import streamlit as st
import joblib
import numpy as np
import pandas as pd
import re
from typing import Optional

st.set_page_config(page_title="Diabetes Prediction System", page_icon="🩺")

# ---------------------------
# Utilities
# ---------------------------
NUM = r"[0-9]+(?:\.[0-9]+)?"

def clean_text(text: str) -> str:
    t = text.strip()
    t = re.sub(r"\s+", " ", t)  # Normalize whitespace
    t = re.sub(r"(?<=\d),(?=\d)", "", t)  # Remove thousand separators
    t = re.sub(rf"({NUM})[^\d\s]", r"\1", t)  # Remove punctuation after numbers
    return t

def find_number(text: str, patterns: list[str]) -> Optional[float]:
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                raw = re.sub(r"[^\d.]+$", "", m.group(1))
                return float(raw)
    return None

def parse_age(text: str) -> Optional[int]:
    pats = [
        rf"\b({NUM})\s*-\s*year\s*-\s*old\b",
        rf"\bAge\s*[:=]\s*({NUM})\b",
        rf"\bI\s*am\s*({NUM})\s*(?:years?|yrs?)\s*old\b",
        rf"\b({NUM})\s*(?:years?|yrs?)\s*old\b",
        rf"\b({NUM})\s*yo\b"
    ]
    val = find_number(text, pats)
    return int(val) if val is not None else None

def parse_bmi(text: str) -> Optional[float]:
    pats = [
        rf"\bBMI\b(?:\s*level)?(?:\s*is)?\s*({NUM})\b",
        rf"\bBMI\s*[:=]\s*({NUM})\b",
    ]
    return find_number(text, pats)

def parse_hba1c(text: str) -> Optional[float]:
    pats = [
        rf"\bHbA1c\b(?:\s*level)?(?:\s*is)?\s*({NUM})\s*%?\b",
        rf"\bA1c\b(?:\s*is)?\s*({NUM})\s*%?\b",
        rf"\bHbA1c\s*[:=]\s*({NUM})\s*%?\b",
    ]
    return find_number(text, pats)

def parse_glucose(text: str) -> Optional[float]:
    pats = [
        rf"\b(?:fasting\s+)?(?:blood\s+)?glucose(?:\s*level)?(?:\s*is)?\s*({NUM})\b",
        rf"\bglucose\s*[:=]\s*({NUM})\b",
        rf"\bFPG\s*[:=]?\s*({NUM})\b",
    ]
    return find_number(text, pats)

def parse_gender(text: str) -> tuple[int, int, int]:
    t = text.lower()
    is_female = bool(re.search(r"\bfemale\b|\bwoman\b", t)) or bool(re.search(r"\bsex\s*[:=]\s*f\b", t))
    is_male   = bool(re.search(r"\bmale\b|\bman\b", t))    or bool(re.search(r"\bsex\s*[:=]\s*m\b", t))
    if is_female and not is_male:
        return 1, 0, 0
    if is_male and not is_female:
        return 0, 1, 0
    return 0, 0, 1  # unclear -> other

def parse_yes_no_flag(text: str, key_word: str) -> Optional[int]:
    t = text.lower()
    if re.search(rf"\b(no|without|denies)\s+{key_word}\b", t):
        return 0
    if re.search(rf"\b{key_word}\s*[:=]\s*(0|no)\b", t):
        return 0
    if re.search(rf"\b{key_word}\b", t):
        if not re.search(rf"\bno\s+{key_word}\b", t):
            return 1
    if re.search(rf"\b{key_word}\s*[:=]\s*(1|yes)\b", t):
        return 1
    return None

SMOKING_CANON = {
    "never": {"never", "non-smoker", "nonsmoker", "does not smoke", "no smoking history", "no tobacco"},
    "former": {"former", "quit", "ex-smoker", "used to smoke"},
    "current": {"current", "smokes", "smoker"},
    "no info": {"no info", "unknown", "unsure"},
    "not current": {"not current"},
    "ever": {"ever"},
}

def parse_smoking(text: str) -> list[int]:
    t = text.lower()
    m = re.search(r"\bsmok(?:e|ing|er)\s*(?:history)?\s*[:=]\s*([a-zA-Z ]+)\b", t)
    label = None
    if m:
        label = m.group(1).strip()
    else:
        for canon, synonyms in SMOKING_CANON.items():
            for s in synonyms:
                if s in t:
                    label = canon
                    break
            if label:
                break
    opts = ["never", "former", "current", "no info", "not current", "ever"]
    if label and label in opts:
        return [int(opt == label) for opt in opts]
    if re.search(r"\b(non[-\s]?smoker|does not smoke|never smoked)\b", t):
        return [1, 0, 0, 0, 0, 0]
    return [0, 0, 0, 1, 0, 0]

def build_df(age, bmi, hba1c, glucose, hypertension, heart_disease,
             gender_female, gender_male, gender_other,
             smoke_never, smoke_former, smoke_current,
             smoke_noinfo, smoke_notcurrent, smoke_ever) -> pd.DataFrame:
    columns = [
        'age', 'bmi', 'HbA1c_level', 'blood_glucose_level',
        'hypertension', 'heart_disease',
        'gender_female', 'gender_male', 'gender_other',
        'smoking_never', 'smoking_former', 'smoking_current',
        'smoking_noinfo', 'smoking_notcurrent', 'smoking_ever'
    ]
    row = [[
        age, bmi, hba1c, glucose,
        hypertension, heart_disease,
        gender_female, gender_male, gender_other,
        smoke_never, smoke_former, smoke_current,
        smoke_noinfo, smoke_notcurrent, smoke_ever
    ]]
    return pd.DataFrame(row, columns=columns)

def safe_flags(text: str):
    htn = parse_yes_no_flag(text, "hypertension")
    hd  = parse_yes_no_flag(text, "heart disease")
    return (0 if htn is None else htn, 0 if hd is None else hd)

# ---------------------------
# Load model
# ---------------------------
@st.cache_resource
def load_model():
    return joblib.load("xgboost_diabetes_model.pkl")

model = load_model()

st.title("Diabetes Prediction System")

# ---------------------------
# Option 1: Manual form
# ---------------------------
st.header("Option 1: Fill out patient information manually")

age = st.number_input("Age", min_value=0, max_value=120, value=30)
bmi = st.number_input("BMI", value=22.5)
hba1c = st.number_input("HbA1c Level", value=5.6)
glucose = st.number_input("Blood Glucose Level", value=100.0)
hypertension = st.selectbox("Hypertension", [0, 1], index=0)
heart_disease = st.selectbox("Heart Disease", [0, 1], index=0)
gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=0)
smoking = st.selectbox("Smoking History", ["never", "former", "current", "No Info", "not current", "ever"], index=0)

gender_female = int(gender == "Female")
gender_male   = int(gender == "Male")
gender_other  = int(gender == "Other")

smoke_never      = int(smoking == "never")
smoke_former     = int(smoking == "former")
smoke_current    = int(smoking == "current")
smoke_noinfo     = int(smoking == "No Info")
smoke_notcurrent = int(smoking == "not current")
smoke_ever       = int(smoking == "ever")

df_manual = build_df(age, bmi, hba1c, glucose, hypertension, heart_disease,
                     gender_female, gender_male, gender_other,
                     smoke_never, smoke_former, smoke_current,
                     smoke_noinfo, smoke_notcurrent, smoke_ever)

if st.button("Predict (Manual)"):
    prob = model.predict_proba(df_manual)[0][1] * 100
    if prob >= 40:
        st.error(f"You are likely diabetic with {prob:.1f}% confidence.")
    else:
        st.success(f"You are not diabetic. Risk is {prob:.1f}%.")

# ---------------------------
# Option 2: Natural language
# ---------------------------
st.markdown("---")
st.header("Option 2: Paste natural language prompt")
st.caption("""Example: "I'm a 30-year-old male. BMI is 25. HbA1c is 5.2. 
Fasting glucose 92 mg/dL. No hypertension or heart disease. Non-smoker.""")

prompt = st.text_area("Enter description")

if st.button("Predict (Natural Language)"):
    try:
        prompt_clean = clean_text(prompt)

        age_n   = parse_age(prompt_clean)
        bmi_n   = parse_bmi(prompt_clean)
        a1c_n   = parse_hba1c(prompt_clean)
        glu_n   = parse_glucose(prompt_clean)

        missing = []
        if age_n is None: missing.append("Age")
        if bmi_n is None: missing.append("BMI")
        if a1c_n is None: missing.append("HbA1c")
        if glu_n is None: missing.append("Glucose")
        if missing:
            raise ValueError(f"Missing or unparsable fields: {', '.join(missing)}. "
                             f"Try formats like 'Age: 30; BMI: 25; HbA1c: 5.2; Glucose: 92'.")

        g_f, g_m, g_o = parse_gender(prompt_clean)
        htn, hd = safe_flags(prompt_clean)
        smoke_vec = parse_smoking(prompt_clean)
        s_never, s_former, s_current, s_noinfo, s_notcurrent, s_ever = smoke_vec

        df_agent = build_df(
            age_n, bmi_n, a1c_n, glu_n, htn, hd,
            g_f, g_m, g_o,
            s_never, s_former, s_current, s_noinfo, s_notcurrent, s_ever
        )

        prob = model.predict_proba(df_agent)[0][1] * 100
        if prob >= 40:
            st.error(f"[Agentic AI] You are likely diabetic with {prob:.1f}% confidence.")
        else:
            st.success(f"[Agentic AI] You are not diabetic. Risk is {prob:.1f}%.")

        with st.expander("Parsed values (debug)"):
            st.json({
                "age": age_n, "bmi": bmi_n, "HbA1c": a1c_n, "glucose": glu_n,
                "hypertension": htn, "heart_disease": hd,
                "gender_female": g_f, "gender_male": g_m, "gender_other": g_o,
                "smoking": {
                    "never": s_never, "former": s_former, "current": s_current,
                    "no info": s_noinfo, "not current": s_notcurrent, "ever": s_ever
                }
            })

    except Exception as e:
        st.warning(f"Could not parse input properly. Error: {e}")

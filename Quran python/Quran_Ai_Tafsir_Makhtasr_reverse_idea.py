import pandas as pd
import streamlit as st
import re
from openai import OpenAI

# Load your Quran with Tafsir dataset
tafsir_df = pd.read_csv("Quran with tafsir.csv")

import os
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def preprocess_text(text):
    return re.sub(r'\s+', ' ', text.strip())

def chat_with_gpt(prompt):
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=600,
        temperature=0.8
    )
    return response.choices[0].text.strip()

def get_tafsir_by_ayah(ayah_en):
    match = tafsir_df[tafsir_df['ayah_en'].str.contains(ayah_en[:30], case=False, na=False)]
    if not match.empty:
        return match.iloc[0]['tafsir']
    return None

def generate_quranic_advice(situation, lang="en"):
    if lang == "en":
        prompt = f"""
Situation: "{situation}"

Find a relevant Quran verse that addresses this situation. Provide the verse, Surah name, and practical advice from the verse. Mention the ayah in English.
"""
    else:
        prompt = f"""
الوضع: "{situation}"

ابحث عن آية من القرآن تتعلق بهذا الموقف، واذكر السورة والآية، ثم قدّم نصيحة عملية مستندة إلى معناها.
"""
    return chat_with_gpt(preprocess_text(prompt))

# Streamlit UI
st.title("🕌 Real-Life to Quran Assistant")

lang = st.radio("Choose Language:", ["English", "Arabic"])
situation = st.text_area("Describe your real-life situation:")

if st.button("Get Quranic Advice"):
    if situation:
        lang_code = "en" if lang == "English" else "ar"
        advice = generate_quranic_advice(situation, lang_code)
        st.success("Quranic Advice:")
        st.write(advice)
    else:
        st.warning("Please describe a situation first.")

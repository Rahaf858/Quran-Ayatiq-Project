import pandas as pd
import re
import streamlit as st
from openai import OpenAI

# Load Quran with Tafsir
tafsir_df = pd.read_csv("Quran with tafsir.csv")

import os
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def preprocess_text(text):
    return re.sub(r'\s+', ' ', text.strip())

def chat_with_gpt(prompt):
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=500,
        temperature=0.8
    )
    return response.choices[0].text.strip()

def get_tafsir(verse, surah):
    match = tafsir_df[
        (tafsir_df['surah_name_roman'].str.lower() == surah.lower()) &
        (tafsir_df['ayah_en'].str.contains(verse, case=False, na=False))
    ]
    if not match.empty:
        return match.iloc[0]['tafsir']
    return "Tafsir not found."

def generate_advice(verse, surah):
    tafsir = get_tafsir(verse, surah)
    prompt = f"""
Quran verse: "{verse}"
Surah: {surah}

Tafsir explanation (Arabic): "{tafsir}"

Based on this, give Three advices that can be applied, and practical behavioral advice to help someone implement it in their life.
"""
    return chat_with_gpt(preprocess_text(prompt))

# Streamlit UI
st.title("🕌 Quran Life Advice Assistant (with Tafsir)")

verse = st.text_area("Enter a Quran verse (in English):")
surah = st.text_input("Enter the Surah name (e.g., Al-Fatihah):")

if st.button("Get Practical Advice"):
    if verse and surah:
        advice = generate_advice(verse, surah)
        st.success("Practical Advice:")
        st.write(advice)
    else:
        st.warning("Please fill in both the verse and the Surah name.")

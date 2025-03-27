import pandas as pd
import re
import streamlit as st
from openai import OpenAI

# ----- Streamlit Page Configuration (FIRST COMMAND) -----
st.set_page_config(page_title="Quran Life Assistant", layout="centered")

# Custom CSS for enhanced styling
st.markdown("""
<style>
    /* Custom color palette */
    :root {
        --primary-color: #2A4D69;
        --secondary-color: #4B86B4;
        --background-light: #F5F5F5;
        --text-color: #333333;
        --accent-color: #D9B26F;
    }

    /* Global styling */
    .stApp {
        background-color: var(--background-light);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Header styling */
    h1 {
        color: var(--primary-color) !important;
        text-align: center;
        font-weight: bold;
        margin-bottom: 20px;
    }

    /* Input and button styling */
    .stTextArea > div > div > textarea {
        background-color: white !important;
        border: 2px solid var(--secondary-color) !important;
        border-radius: 10px !important;
        color: var(--text-color) !important;
    }

    .stButton > button {
        background-color: var(--primary-color) !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: bold !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button:hover {
        background-color: var(--secondary-color) !important;
        transform: scale(1.05) !important;
    }

    /* Result boxes */
    .guidance-box {
        background-color: rgba(75, 134, 180, 0.1) !important;
        border-left: 5px solid var(--primary-color) !important;
        padding: 15px !important;
        border-radius: 10px !important;
        margin-top: 15px !important;
    }

    .tafsir-box {
        background-color: rgba(217, 178, 111, 0.1) !important;
        border-left: 5px solid var(--accent-color) !important;
        padding: 15px !important;
        border-radius: 10px !important;
        margin-top: 15px !important;
        direction: rtl !important;
    }
</style>
""", unsafe_allow_html=True)

# ----- Load Quran CSV -----
df = pd.read_csv("The Quran Dataset.csv")
columns_to_keep = [
    'ayah_en', 'ayah_ar', 'place_of_revelation',
    'surah_name_en', 'surah_name_ar',
    'no_of_word_ayah', 'list_of_words'
]
df = df[columns_to_keep].dropna().reset_index().rename(columns={'index': 'ayah_index'})

# ----- Load and Parse Tafsir Markdown -----
def parse_tafsir_saddi_markdown(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    entries = re.findall(r'#\s*(\d+)\n\n(.*?)(?=\n#\s*\d+|\Z)', content, re.DOTALL)
    saddi_data = [{'ayah_index': int(idx), 'tafsir_ar': tafsir.strip()} for idx, tafsir in entries]
    return pd.DataFrame(saddi_data)

saddi_df = parse_tafsir_saddi_markdown("ar-al-saddi-qurancom.md")
merged_df = df.merge(saddi_df, on='ayah_index', how='left')

import os
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def preprocess_text(text):
    return re.sub(r'\s+', ' ', text.strip())

def chat_with_gpt(prompt):
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that gives Quran-based life advice."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=800,
        temperature=0.8
    )
    return response.choices[0].message.content.strip()

def generate_response(situation, lang):
    if lang == "English":
        prompt = f"""
Real-life situation: "{situation}"

1. Suggest a relevant Quran verse (English) and Surah name that gives guidance.
2. Explain practical advice based on that verse for this situation.
"""
    else:
        prompt = f"""
الموقف الواقعي: "{situation}"

١. اقترح آية من القرآن (بالعربية) واسم السورة التي تقدم التوجيه في هذا الموقف.
٢. قدم نصيحة عملية مبنية على هذه الآية.
"""
    return chat_with_gpt(preprocess_text(prompt))

def get_tafsir_by_ayah(ayah_text):
    match = merged_df[merged_df['ayah_en'].str.contains(ayah_text[:30], case=False, na=False)]
    if not match.empty:
        return match.iloc[0]['tafsir_ar']
    return None

# ----- Main App -----
st.markdown("<h1>🕌 Quran Life Guidance Assistant</h1>", unsafe_allow_html=True)

# Language Choice
lang = st.radio("🌐 Select language / اختر اللغة:", ["English", "Arabic"])
situation = st.text_area("📝 Describe your real-life situation / صف الموقف الذي تمر به:")

if st.button("📖 Get Quranic Guidance"):
    if situation:
        with st.spinner("🔎 Searching the Quran for relevant guidance..."):
            response = generate_response(situation, lang)

        st.markdown("### 🔹 Quranic Guidance and Advice")
        st.markdown(f'<div class="guidance-box">{response}</div>', unsafe_allow_html=True)

        if lang == "English":
            tafsir = get_tafsir_by_ayah(response)
            if tafsir:
                st.markdown("### 📖 Tafsir Al-Sa'di (Arabic)")
                st.markdown(f'<div class="tafsir-box">{tafsir}</div>', unsafe_allow_html=True)
            else:
                st.info("Tafsir not found for the matched verse.")
    else:
        st.warning("Please describe your situation first.")
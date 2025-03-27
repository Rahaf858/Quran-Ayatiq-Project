import pandas as pd
import time
import re
from openai import OpenAI
import streamlit as st


import pandas as pd

df = pd.read_csv('The Quran Dataset.csv')


# Keep only selected columns
columns_to_keep = [
    'ayah_en', 
    'ayah_ar',
    'place_of_revelation',
    'surah_name_en',
    'surah_name_ar',
    'no_of_word_ayah',
    'list_of_words'
]
df = df[columns_to_keep]

# Drop rows with missing values
df.dropna(inplace=True)

import os
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def preprocess_text(text):
    # Optional: add preprocessing here
    return re.sub(r'\s+', ' ', text.strip())

def chat_with_gpt(message):
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=message,
        max_tokens=300,
        temperature=0.8
    )
    return response.choices[0].text.strip()




def generate_advice_en(verse_en, surah_en):
    prompt = f"""Quran verse: "{verse_en}" (Surah: {surah_en})

Provide a real-life situation where this verse can be applied, and give practical behavioral advice to help someone implement it."""
    return chat_with_gpt(preprocess_text(prompt))

def generate_advice_ar(verse_ar, surah_ar):
    prompt = f"""الآية: "{verse_ar}" (السورة: {surah_ar})

اكتب موقفًا واقعيًا يعكس معنى هذه الآية، وقدم نصيحة سلوكية عملية تساعد المسلم على تطبيقها في حياته اليومية."""
    return chat_with_gpt(preprocess_text(prompt))


#Chating in the proment
#while True:
    #if user_input.lower() == "quit":
    #break
    #surah_input = input("Enter the Surah name: ")
    #prompt = f"""Quran verse: "{user_input}" (Surah: {surah_input})

#Give a real-life situation and practical advice to apply this verse."""
    #print("AI:", chat_with_gpt(prompt))




st.title("Quran Life Advice Assistant 🤖📖")

verse = st.text_area("Enter a Quran verse (in English or Arabic):")
surah = st.text_input("Enter the Surah name:")

if st.button("Get Practical Advice"):
    if verse and surah:
        prompt_en = f"""Quran verse: "{verse}" (Surah: {surah})

Give a real-life situation and practical advice to apply this verse."""
        advice = chat_with_gpt(prompt_en)
        st.success("Practical Advice:")
        st.write(advice)




#Ues  versue translate
# could give situation then based on it give there advice
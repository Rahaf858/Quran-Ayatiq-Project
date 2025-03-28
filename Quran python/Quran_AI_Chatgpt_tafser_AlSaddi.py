import os
import streamlit as st
from langdetect import detect
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# 🧠 OpenAI API Key
st.set_page_config(page_title="Raqaiq - Quran AI Assistant", layout="centered")
st.title("📖 رقائق - Quran Life Advice")
openai_key = st.text_input("🔐 Enter your OpenAI API key", type="password")
os.environ["OPENAI_API_KEY"] = openai_key

# 📘 Upload tafsir file (if needed)
if "vectorstore" not in st.session_state:
    with st.spinner("🔍 Loading Tafsir..."):
        loader = TextLoader("ar-al-saddi-qurancom.md", encoding="utf-8")
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
        st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)

# 🌐 Language detection helper
def detect_language(text):
    try:
        lang = detect(text)
        return "ar" if lang == "ar" else "en"
    except:
        return "en"

# 💬 Language prompts
PROMPTS = {
    "ar": {
        "verse": "اقترح آية قرآنية تناسب هذا الموقف: {situation}",
        "advice": "أعط نصيحة عملية لشخص يمر بهذا الموقف: {situation}\nواستند على هذه الآية: {verse}"
    },
    "en": {
        "verse": "Suggest a Quranic verse that fits this situation: {situation}",
        "advice": "Give practical advice to someone facing this situation: {situation}\nBased on this verse: {verse}"
    }
}

# ✍️ User input
situation = st.text_area("💬 Describe your situation (in Arabic or English):")

if st.button("🔎 Get Quranic Guidance") and situation:
    lang = detect_language(situation)
    st.markdown(f"🌍 Detected language: **{'Arabic' if lang == 'ar' else 'English'}**")

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)

    verse_prompt = PromptTemplate(input_variables=["situation"], template=PROMPTS[lang]["verse"])
    advice_prompt = PromptTemplate(input_variables=["situation", "verse"], template=PROMPTS[lang]["advice"])

    verse_chain = LLMChain(llm=llm, prompt=verse_prompt)
    advice_chain = LLMChain(llm=llm, prompt=advice_prompt)

    retriever = st.session_state.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    with st.spinner("🧠 Generating verse..."):
        verse = verse_chain.run(situation).strip()
        st.markdown(f"### 📜 Quranic Verse:\n{verse}")

    with st.spinner("📘 Retrieving Tafsir..."):
        tafsir = qa.run(verse)
        st.markdown("### 📖 Tafsir Al-Sa'di:")
        st.write(tafsir)

    with st.spinner("💡 Generating advice..."):
        advice = advice_chain.run({"situation": situation, "verse": verse})
        st.markdown("### 💡 Practical Advice:")
        st.write(advice)

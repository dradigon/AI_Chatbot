import streamlit as st
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from gtts import gTTS
import tempfile
import os

# Load stopwords for filtering
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def extract_keywords(text):
    words = text.lower().split()
    keywords = [word for word in words if word not in stop_words]
    return " ".join(keywords) if keywords else text

@st.cache_data
def load_medical_dataset():
    dataset = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train")
    return dataset['input'], dataset['output']

@st.cache_resource
def load_sbert():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_clinical_bert():
    model_name = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).half()
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return tokenizer, model

@st.cache_resource
def compute_question_embeddings(questions):
    sbert_model = load_sbert()
    return sbert_model.encode(questions, convert_to_tensor=True)

questions, answers = load_medical_dataset()
sbert_model = load_sbert()
question_embeddings = compute_question_embeddings(questions)

def get_best_match_sbert(user_query):
    query_embedding = sbert_model.encode(user_query, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, question_embeddings)
    best_idx = similarities.argmax().item()
    if similarities[0, best_idx] < 0.7:
        return None, "Sorry, I don’t have an answer for that.", 0
    return questions[best_idx], answers[best_idx], similarities[0, best_idx].item() * 100

def refine_answer(question, raw_answer, tokenizer, model):
    input_text = f"A patient asked: {question}. The original response was: {raw_answer}. Please improve this answer."
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(model.device)
    output = model.generate(**inputs, max_length=200)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        tts.save(temp_audio.name)
        return temp_audio.name

st.title("🩺 Medical Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

tokenizer, model = load_clinical_bert()

user_query = st.chat_input("Ask a medical question...")
if user_query:
    with st.spinner("Processing query..."):
        processed_query = extract_keywords(user_query)
        match, raw_answer, score = get_best_match_sbert(processed_query)
        refined_answer = refine_answer(match, raw_answer, tokenizer, model) if match else raw_answer

    st.session_state.chat_history.append(("user", user_query))
    st.session_state.chat_history.append(("bot", refined_answer))

    with st.chat_message("bot"):
        st.write(refined_answer)
        audio_file = text_to_speech(refined_answer)
        st.audio(audio_file, format="audio/mp3")
        os.remove(audio_file)

for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(msg)

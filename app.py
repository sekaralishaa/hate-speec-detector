#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
import streamlit as st
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle
import joblib

# Custom CSS
st.markdown(
    """
    <style>
    .title {
        color: #000080;  /* Navy */
        font-size: 2.5em;
        font-weight: bold;
    }
    .answer-box {
        background-color: #000080;  /* Navy */
        color: white;
        font-size: 1.2em;
        padding: 15px;
        border-radius: 5px;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Streamlit app
st.markdown('<div class="title">Hate Speech and Offensive Language Detection</div>', unsafe_allow_html=True)

# Input text box
user_input = st.text_area("Enter a sentence to classify:")

if st.button("Predict"):
    if user_input:
        clean_text = cleansing(user_input)
        vectorized_text = vectorizer_cvec.transform([clean_text])
        prediction = random_forest.predict(vectorized_text)[0]
        class_map = {0: "Hate Speech", 1: "Offensive Language", 2: "Neutral"}
        
        # Display prediction in a styled box
        st.markdown(
            f'<div class="answer-box">The text is classified as: <strong>{class_map[prediction]}</strong></div>',
            unsafe_allow_html=True,
        )
    else:
        st.warning("Please enter a sentence.")


# Load pre-trained model and vectorizer
with open("random_forest_model_compressed.pkl", "rb") as model_file:
    random_forest = joblib.load("random_forest_model_compressed.pkl")
    # random_forest = pickle.load(model_file)
with open("count_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer_cvec = pickle.load(vectorizer_file)

def cleansing(text):
    # Preprocess the input text
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = re.sub(url_pattern, '', text)
    text = re.sub(r"\d+", "", text.lower())
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    stop_words = set(stopwords.words('english'))
    tokenized_text = word_tokenize(text)
    tokenized_text = [word for word in tokenized_text if word not in stop_words]
    stemmer = PorterStemmer()
    tokenized_text = [stemmer.stem(word) for word in tokenized_text]
    lemmatizer = WordNetLemmatizer()
    tokenized_text = [lemmatizer.lemmatize(word) for word in tokenized_text]
    return ' '.join(tokenized_text)

# # Streamlit app
# st.markdown('<div class="title">Hate Speech and Offensive Language Detection</div>', unsafe_allow_html=True)

# # Input text box
# user_input = st.text_area("Enter a sentence to classify:")

# if st.button("Predict"):
#     if user_input:
#         clean_text = cleansing(user_input)
#         vectorized_text = vectorizer_cvec.transform([clean_text])
#         prediction = random_forest.predict(vectorized_text)[0]
#         class_map = {0: "Hate Speech", 1: "Offensive Language", 2: "Neutral"}
        
#         # Display prediction in a styled box
#         st.markdown(
#             f'<div class="answer-box">The text is classified as: <strong>{class_map[prediction]}</strong></div>',
#             unsafe_allow_html=True,
#         )
#     else:
#         st.warning("Please enter a sentence.")

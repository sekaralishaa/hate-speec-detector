#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load pre-trained model and vectorizer
with open("random_forest_model_compressed.pkl", "rb") as model_file:
    random_forest = pickle.load(model_file)
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

# Streamlit app
st.title("Hate Speech and Offensive Language Detection")

# Input text box
user_input = st.text_area("Enter a sentence to classify:")

if st.button("Predict"):
    if user_input:
        clean_text = cleansing(user_input)
        vectorized_text = vectorizer_cvec.transform([clean_text])
        prediction = random_forest.predict(vectorized_text)[0]
        class_map = {0: "Hate Speech", 1: "Offensive Language", 2: "Neutral"}
        st.write(f"The text is classified as: **{class_map[prediction]}**")
    else:
        st.warning("Please enter a sentence.")


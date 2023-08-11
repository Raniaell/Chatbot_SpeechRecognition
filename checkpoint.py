import nltk
import streamlit as st
import speech_recognition as sr
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

with open('geology.txt', 'r', encoding='utf-8') as f:
    data = f.read().replace('\n', ' ')

sentences = sent_tokenize(data) #Tokenize the text into sentences

def preprocess(sentence):
    words = word_tokenize(sentence) #Tokenize the sentence into words
    words=[word.lower() for word in words if word.lower() not in stopwords.words('english') and word not in string.punctuation ]
    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return words

corpus = [preprocess(sentence) for sentence in sentences]

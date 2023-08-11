
import speech_recognition as sr
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import streamlit as st 

# Initialize the speech recognition recognizer
r = sr.Recognizer()

with open('geology.txt', 'r', encoding='utf-8') as f:
    data = f.read().replace('\n', ' ')

# Tokenize the text into sentences
sentences = sent_tokenize(data)
# Define a function to preprocess each sentence
def preprocess(sentence):
    # Tokenize the sentence into words
    words = word_tokenize(sentence)
    # Remove stopwords and punctuation
    words = [word.lower() for word in words if word.lower() not in stopwords.words('english') and word not in string.punctuation]
    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return words

# Preprocess each sentence in the text
corpus = [preprocess(sentence) for sentence in sentences]

# Define a function to find the most relevant sentence given a query
def get_most_relevant_sentence(query):
    # Preprocess the query
    query = preprocess(query)
    # Compute the similarity between the query and each sentence in the text
    max_similarity = 0
    most_relevant_sentence = ""
    for sentence in corpus:
        similarity = len(set(query).intersection(sentence)) / float(len(set(query).union(sentence)))
        if similarity > max_similarity:
            max_similarity = similarity
            most_relevant_sentence = " ".join(sentence)
    return most_relevant_sentence
def chatbot(question):
    # Find the most relevant sentence
    most_relevant_sentence = get_most_relevant_sentence(question)
    # Return the answer
    return most_relevant_sentence


    
def transcribe_speech():
    with sr.Microphone() as source:
        st.info('Speak now...')
        print("This is only available for english speakers! Thanks for understanding!")
        audio_data = r.record(source, duration=5)
        print("Recognizing...")
        print("this is your recording: ",audio_data)
        try:
            text = r.recognize_google(audio_data)
            print(audio_data.get_wav_data(), text)
            st.write("Transcription:", text)
            return text
        except:
            print("Sorry I didn't get that or you run out of time")
            st.write("Sorry I didn't get that/You run out of time")

def main():
    st.title("Chatbot with Text/Speech Input")

    # Choose input type: Text or Speech
    input_type = st.radio("Choose input type:", ["Text", "Speech"])

    user_input = ""
    if input_type == "Text":
        user_input = st.text_input("User:")
    elif input_type == "Speech":
        user_input = transcribe_speech()

    if user_input:
        response = chatbot(user_input)
        st.write("Chatbot:", response)

if __name__ == "__main__":
    main()


# to run this script type in terminal streamlit run main.py

# Imports are written here
import re
import pickle
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

# Setting up streamlit
st.title("Sarcasm Detector")
st.balloons()


def preprocess_signs(input_text):
    """
    :param input_text: the input text received from the user
    :return: text: preprocessed text
    """

    # Lowering the text
    input_text = input_text.lower()

    # Remove URLs
    input_text = re.sub(r'http\S+', '', input_text)

    # Remove mentions and hashtags
    input_text = re.sub(r'@\w+|#\w+', '', input_text)

    # Remove extra whitespace
    input_text = re.sub('\s+', ' ', input_text).strip()

    # Tf-IDF Vectorizer
    vectorizer = TfidfVectorizer()
    input_text = vectorizer.fit_transform([input_text])

    return input_text


def model_predict(input_text):
    """
    :param input_text: preprocessed input text
    :return:
    """
    try:
        # Loading the model
        with open('Model/model.pkl', 'rb') as f:
            model = pickle.load(f)

        # Prediction on text
        result = model.predict(input_text)

        # If model predicts 1 it means Headline contains Sarcasm else it's not present
        if result[0] == 1:
            return True
        else:
            return False

    except Exception as e:
        raise Exception(e)


# Getting the text from user
text = st.text_input('Enter the headline....')

if text:
    text = preprocess_signs(input_text=text)
    model_predict(input_text=text)

# Credits:
with st.sidebar:
    st.header("About this project:")
    st.write("The problem statement for the sarcasm detection NLP project using logistic regression for headlines is "
             "to develop an accurate and reliable model that can automatically detect whether a given headline is "
             "sarcastic or not. With the vast amount of textual data available online, it is becoming increasingly "
             "important to be able to accurately distinguish between sarcastic and non-sarcastic content. This can be "
             "useful in various applications such as social media monitoring, online reputation management, "
             "and sentiment analysis. The main goal of this project is to provide a solution that can efficiently and "
             "effectively identify sarcastic headlines, thereby improving the accuracy and reliability of NLP "
             "applications.")
    st.header("Created by Meetttttt")
    st.header("For More Information Check out this [GitHub](https://github.com/meetttttt/Sarcasm-Detector)")

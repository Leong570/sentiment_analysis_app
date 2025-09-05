import streamlit as st #pulls in Streamlit for UI building
import joblib #allow loading of .pkl files saved from training
from langdetect import detect #function that guesses the language code from a text string
from deep_translator import GoogleTranslator #lightweight translator to convert non-English text to English
import re

def handle_negation(text):
    negation_words = {"not", "no", "never"}
    text = re.sub(r"\b(\w+)(n't)\b", r"\1 not", text)

    tokens = text.split()
    result = []
    skip_next = False

    for token in tokens:
        lower = token.lower()
        if skip_next:
            result.append("not_" + token)
            skip_next = False
        elif lower in negation_words:
            skip_next = True
        else:
            result.append(token)
    return " ".join(result)

#Load the models and TF-IDF vectorizer
#load the same TF-IDF vectorizer that is fit during training
tfidf = joblib.load("tfidf_vectorizer.pkl")

#map the name to each sentiment classifier
models = {
    "Complement Naive Bayes": joblib.load("cnb_model.pkl"),
    "Logistic Regression": joblib.load("lg_model.pkl"),
    "Support Vector Machine": joblib.load("svm_model.pkl")
}

#streamlit UI
st.title("🎬🍿Movie Review Sentiment Analyser")

#text field for user to input movie review
user_review = st.text_area("Enter a movie review: ")

#render a button and add a listener to handle
if st.button("Analyse and Predict the Sentiment"):
    #check if the text is empty
    if user_review.strip() == "":
        st.warning("Please enter a review before analysing or predicting.")
    else:
        #Detect language as the first step
        try:
            lang = detect(user_review)
        except:
            lang = "unknown" #unable to detect the language
        
        st.write(f"Detected Language: {lang}")

        #translate if not English as second step
        if lang != "en":
            translator = GoogleTranslator(source="auto", target="en") #auto --> let the translator auto-detect
            review_english = translator.translate(user_review)
            st.write(f"Translated to English: {review_english}")
        else:
            review_english = user_review

        #preprocess the review given by user as third step
        review_processed = handle_negation(review_english) 
        review_vector = tfidf.transform([review_processed])

        #prediction using chosen model as fourth step
        for model_name, model in models.items():
            sentiment = model.predict(review_vector)[0]

            #display the sentiment as final step
            if sentiment == 1:
                st.success(f"{model_name}'s Prediction: Positive 😁")
            else:
                st.error(f"{model_name}'s Prediction: Negative 🥲")

        st.write("The application can make mistake. Check before you take the result.")

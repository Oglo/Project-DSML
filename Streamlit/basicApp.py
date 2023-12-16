import streamlit as st
from joblib import load
import requests
from io import BytesIO
from PIL import Image
import spacy
import pandas as pd
import joblib

# Fonction pour charger le modèle depuis GitHub
def load_model(url):
    response = requests.get(url)
    model = load(BytesIO(response.content))
    return model

# Fonction pour extraire les caractéristiques du texte
def extract_features(text):
    nlp = spacy.load('fr_core_news_sm')
    doc = nlp(text)
    num_sentences = len(list(doc.sents))
    num_tokens = len(doc)
    avg_sentence_length = num_tokens / num_sentences if num_sentences > 0 else 0
    lexical_diversity = len(set(token.text.lower() for token in doc)) / num_tokens if num_tokens > 0 else 0
    return [avg_sentence_length, lexical_diversity]

# Fonction pour prédire avec Logistic Regression
def predict1(model, text):
    features = extract_features(text)
    features_df = pd.DataFrame([features], columns=['avg_sentence_length', 'lexical_diversity'])
    prediction = model.predict(features_df)
    return prediction

# Fonction pour prédire avec SVC
def predict2(model, vectorizer, text):
    processed_text = vectorizer.transform([text])
    prediction = model.predict(processed_text)
    return prediction

# Interface Streamlit
logo_url = "https://raw.githubusercontent.com/Oglo/Project-DSML/main/Code/images/logomigros.png"

# Télécharger l'image depuis l'URL
response = requests.get(logo_url)
logo_img = Image.open(BytesIO(response.content))

# Création d'une structure de colonnes pour aligner le titre et l'image
col1, col2, col3 = st.columns([1, 2, 1])

with col2:  # Utilisation de la colonne centrale
    st.markdown("<h1 style='text-align: center'>Team</h1>", unsafe_allow_html=True)    
    st.image(logo_img)

# Sélection du pourcentage de précision
precision = st.selectbox('Choisissez le pourcentage de précision :', ['30%', '40%', '45%', '55%', '60%'])

# Charger le modèle et le vectorizer si nécessaire
model_choice = None
if precision == '30%':
    models = ['Random Forest (36%)', 'Spacy (32%)']
elif precision == '45%':
    models = ['Logistic Regression (45%)', 'RNN']
elif precision == '40%':
    models = ['Vector']
# Ajoutez d'autres conditions pour les autres pourcentages

model_choice = st.selectbox('Choisissez un modèle :', models)

# Zone de texte pour la saisie de la phrase
user_input = st.text_area("Entrez votre texte ici:")

# Chargement et prédiction du modèle
if st.button(f'Prédire le niveau de langue avec {model_choice}'):
    if 'Logistic Regression (45%)' in model_choice:
        model_url = f"https://github.com/Oglo/Project-DSML/raw/main/Streamlit/{model_choice}.joblib"
        model = load_model(model_url)
        prediction = predict1(model, user_input)
    elif 'Vector' in model_choice:
        model_url = f"https://github.com/Oglo/Project-DSML/raw/main/Streamlit/{model_choice}.joblib"
        vectorizer_url = f"https://github.com/Oglo/Project-DSML/raw/main/Streamlit/vectorizer.joblib"  # Assurez-vous que cette URL est correcte
        model = load_model(model_url)
        vectorizer = joblib.load(BytesIO(requests.get(vectorizer_url).content))
        prediction = predict2(model, vectorizer, user_input)
    elif 'Spacy (32%)' in model_choice:
        model_url = f"https://github.com/Oglo/Project-DSML/raw/main/Streamlit/{model_choice}.joblib"
        model = load_model(model_url)
        prediction = predict1(model, user_input)
    # Ajoutez d'autres conditions pour les autres modèles si nécessaire

    # Dictionnaire pour mapper les chiffres aux niveaux de langue
    niveau_langue = {0: 'A1', 1: 'A2', 2: 'B1', 3: 'B2', 4: 'C1', 5: 'C2'}

    # Affichage du niveau de langue correspondant
    predicted_level = niveau_langue[prediction[0]]
    st.write(f'Niveau de langue prédit: {predicted_level}')

import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import joblib
import pickle


# Téléchargement du modèle depuis GitHub
model_url = "https://github.com/Oglo/Project-DSML/raw/main/Streamlit/model_langue.pkl"
response = requests.get(model_url)
model_file = BytesIO(response.content)
model = pickle.load(model_file)

# Création d'un champ de saisie pour le texte à analyser
user_text = st.text_area("Entrez votre texte ici :")

# Bouton pour effectuer la prédiction
if st.button('Prédire le niveau de langue'):
    if user_text:
        # Prédiction
        predicted_level = model.predict([user_text])
        # Traduction du niveau de langue en termes compréhensibles
        level_mapping = {0: 'A1', 1: 'A2', 2: 'B1', 3: 'B2', 4: 'C1', 5: 'C2'}
        predicted_level_text = level_mapping[predicted_level[0]]
        st.write(f"Le niveau de langue prédit pour le texte est : {predicted_level_text}")
    else:
        st.write("Veuillez entrer un texte.")

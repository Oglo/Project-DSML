
import streamlit as st
import requests
import pickle
from io import BytesIO

# Fonction pour charger le modèle et l'encodeur depuis GitHub
def load_model_from_github(url):
    response = requests.get(url)
    model_file = BytesIO(response.content)
    return pickle.load(model_file)

# Liens GitHub
model_url = 'https://raw.githubusercontent.com/Oglo/Project_DSML/main/Streamlit/model_pipeline.pkl'
label_encoder_url = 'https://raw.githubusercontent.com/Oglo/Project_DSML/main/Streamlit/label_encoder (1).pkl'

# Charger le modèle et l'encodeur
model = load_model_from_github(model_url)
label_encoder = load_model_from_github(label_encoder_url)

# Interface Streamlit
st.title("Prédiction du niveau de langue")

# Zone de texte pour l'entrée de l'utilisateur
user_input = st.text_area("Entrez votre texte ici:")

if st.button('Prédire'):
    # Prédiction
    prediction = model.predict([user_input])
    prediction_label = label_encoder.inverse_transform(prediction)
    
    # Afficher la prédiction
    st.write(f"Niveau de langue prédit : {prediction_label[0]}")


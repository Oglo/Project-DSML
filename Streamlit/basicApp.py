
import streamlit as st
import requests
import pickle

# Télécharger le modèle depuis GitHub
@st.cache
def load_model(url):
    response = requests.get(url)
    model = pickle.loads(response.content)
    return model

# Remplacer par l'URL de votre modèle sur GitHub
model_url = 'https://github.com/Oglo/Projec-DSML/raw/main/Streamlit/language_level_model2.pkl'
model = load_model(model_url)

# Création de l'interface utilisateur
st.title('Prédiction du niveau de langue d’un texte en français')

text = st.text_area("Entrez votre texte ici:", "")

if st.button('Prédire le niveau de langue'):
    prediction = model.predict([text])[0]
    st.write(f'Le niveau de langue prédit pour ce texte est : {prediction}')

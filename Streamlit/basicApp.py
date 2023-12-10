import streamlit as st
import requests
import pickle
from io import BytesIO

# URL du modèle sur GitHub
url_modele = 'https://github.com/Oglo/Projec-DSML/raw/main/Streamlit/mon_modele_langue.pkl'

# Fonction pour télécharger et charger le modèle
def charger_modele(url):
    response = requests.get(url)
    modele = BytesIO(response.content)
    return pickle.load(modele)

modele_charge = charger_modele(url_modele)

# Fonction pour faire des prédictions
def predire_niveau(texte):
    return modele_charge.predict([texte])[0]

# Interface Streamlit
st.title('Prédicteur de niveau de langue française')

texte_utilisateur = st.text_area("Entrez votre texte ici:")

if st.button('Prédire'):
    niveau_predi = predire_niveau(texte_utilisateur)
    st.write(f'Le niveau de langue prédit est : {niveau_predi}')

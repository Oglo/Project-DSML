import streamlit as st
import requests
import joblib
from io import BytesIO

# URL du modèle sur GitHub
url_modele = 'https://github.com/Oglo/Project-DSML/raw/main/Streamlit/mon_modele_langue.joblib'

# Fonction pour télécharger et charger le modèle
def charger_modele(url):
    response = requests.get(url)
    if response.status_code == 200:
        modele = BytesIO(response.content)
        return joblib.load(modele)
    else:
        st.error(f"Échec du téléchargement du modèle : statut {response.status_code}")
        return None

# Télécharge et charge le modèle
modele_charge = charger_modele(url_modele)

# Vérifie si le modèle a été chargé avec succès
if modele_charge is not None:

    # Fonction pour faire des prédictions
    def predire_niveau(texte):
        return modele_charge.predict([texte])[0]

    # Interface Streamlit
    st.title('Prédicteur de niveau de langue française')

    texte_utilisateur = st.text_area("Entrez votre texte ici:")

    if st.button('Prédire'):
        niveau_predi = predire_niveau(texte_utilisateur)
        st.write(f'Le niveau de langue prédit est : {niveau_predi}')
else:
    st.error("Le modèle n'a pas pu être chargé. Veuillez vérifier l'URL du modèle.")

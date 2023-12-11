import streamlit as st
import joblib

# Charger le modèle
modele = joblib.load('"C:\\Users\\antoi\\OneDrive\\Bureau\\mon_modele_entraine.pkl"') # Remplacez avec le chemin correct

def predire_niveau(texte):
    # Cette fonction devra être adaptée selon la façon dont votre modèle fonctionne
    prediction = modele.predict([texte])
    return prediction[0]

# Créer l'interface Streamlit
st.title('Prédiction du niveau de langue')

# Zone de texte pour l'entrée de l'utilisateur
texte_utilisateur = st.text_area("Entrez votre texte ici:")

# Bouton de prédiction
if st.button('Prédire'):
    niveau_langue = predire_niveau(texte_utilisateur)
    st.write(f'Le niveau de langue prédit est : {niveau_langue}')

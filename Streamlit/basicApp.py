import streamlit as st
import pickle

# Charger le modèle
with open('mon_modele_langue.pkl', 'rb') as file:
    modele_charge = pickle.load(file)

def predire_niveau(texte):
    return modele_charge.predict([texte])[0]

# Interface Streamlit
st.title('Prédicteur de niveau de langue française')

texte_utilisateur = st.text_area("Entrez votre texte ici:")

if st.button('Prédire'):
    niveau_predi = predire_niveau(texte_utilisateur)
    st.write(f'Le niveau de langue prédit est : {niveau_predi}')

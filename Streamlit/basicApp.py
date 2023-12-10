import streamlit as st
import requests
import pickle
import tempfile
import os

# Télécharger le modèle depuis GitHub et le charger
@st.cache(allow_output_mutation=True)
def load_model(url):
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        return None
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
        for chunk in response.iter_content(chunk_size=8192):
            tmp.write(chunk)
        tmp.flush()
        with open(tmp.name, 'rb') as f:
            model = pickle.load(f)
    os.remove(tmp.name)  # Supprimer le fichier temporaire
    return model

model_url = 'https://github.com/Oglo/Project-DSML/raw/main/Streamlit/language_level_model2.pkl'
model = load_model(model_url)

if not model:
    st.error("Erreur lors du chargement du modèle. Veuillez vérifier l'URL ou réessayer plus tard.")
else:
    # Interface utilisateur Streamlit
    st.title('Prédiction du niveau de langue d’un texte en français')
    text = st.text_area("Entrez votre texte ici:", "")

    if st.button('Prédire le niveau de langue'):
        prediction = model.predict([text])[0]
        st.write(f'Le niveau de langue prédit pour ce texte est : {prediction}')

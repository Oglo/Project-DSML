import streamlit as st
import requests
from joblib import load
import tempfile

# Fonction pour télécharger le modèle depuis GitHub
def download_model(url):
    r = requests.get(url, allow_redirects=True)
    if r.status_code == 200:
        f = tempfile.NamedTemporaryFile(delete=False, suffix='.joblib')
        f.write(r.content)
        f.close()
        return f.name
    else:
        return None

# URL du modèle sur GitHub
model_url = 'https://github.com/Oglo/Project-DSML/raw/main/Streamlit/language_level_classifier.joblib'

# Télécharger le modèle
model_path = download_model(model_url)
if model_path:
    model = load(model_path)
else:
    st.error("Erreur lors du téléchargement du modèle")
    st.stop()

# Interface Streamlit
st.title("Prédiction du Niveau de Langue Française")
text = st.text_area("Entrez votre texte ici:")

if st.button('Prédire'):
    if text:
        prediction = model.predict([text])[0]
        st.success(f"Le niveau de langue prédit est: {prediction}")
    else:
        st.error("Veuillez entrer un texte.")

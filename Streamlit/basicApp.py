import streamlit as st
import requests
import pickle
import tempfile

# Télécharger le modèle depuis GitHub et le charger
@st.cache
def load_model(url):
    response = requests.get(url)
    with tempfile.NamedTemporaryFile(suffix='.pkl') as tmp:
        tmp.write(response.content)
        tmp.flush()
        model = pickle.load(open(tmp.name, 'rb'))
    return model

model_url = 'https://github.com/Oglo/Project-DSML/raw/main/Streamlit/language_level_model2.pkl'
model = load_model(model_url)

# Interface utilisateur Streamlit
st.title('Prédiction du niveau de langue d’un texte en français')
text = st.text_area("Entrez votre texte ici:", "")

if st.button('Prédire le niveau de langue'):
    prediction = model.predict([text])[0]
    st.write(f'Le niveau de langue prédit pour ce texte est : {prediction}')

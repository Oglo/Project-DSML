import streamlit as st
import requests
from joblib import load
import joblib
import sklearn
import tempfile
import requests
from io import BytesIO

# Fonction pour charger le modèle depuis GitHub
def load_model(url):
    response = requests.get(url)
    model = joblib.load(BytesIO(response.content))
    return model

# Interface Streamlit
st.title('Prédiction du niveau de langue')

# Sélection du pourcentage de précision
precision = st.selectbox('Choisissez le pourcentage de précision :', 
                         ['30%', '40%', '50%', '55%', '60%'])

# Afficher les modèles en fonction de la précision choisie
# Cela dépend de la manière dont vous associez les pourcentages aux modèles
if precision == '30%':
    models = ['Logistic_Regression', 'Modèle B']  # Exemple
elif precision == '40%':
    models = ['Modèle C', 'Modèle D']  # Exemple
# Ajoutez d'autres conditions pour les autres pourcentages

model_choice = st.selectbox('Choisissez un modèle :', models)

# Zone de texte pour la saisie de la phrase
user_input = st.text_area("Entrez votre texte ici:")

# Chargement et prédiction du modèle
if st.button('Prédire le niveau de langue'):
    model_url = f"https://github.com/Oglo/Project-DSML/raw/main/Streamlit/{model_choice}.joblib"
    model = load_model(model_url)
    prediction = model.predict([user_input])
    st.write(f'Niveau de langue prédit: {prediction[0]}')
import streamlit as st
import requests
from joblib import load
import joblib
import sklearn
import tempfile
import requests
from io import BytesIO
from PIL import Image


# Fonction pour charger le modèle depuis GitHub
def load_model(url):
    response = requests.get(url)
    model = joblib.load(BytesIO(response.content))
    return model

# Interface Streamlit
logo_url = "https://raw.githubusercontent.com/Oglo/Project-DSML/main/Code/images/logomigros.png"

# Télécharger l'image depuis l'URL
response = requests.get(logo_url)
logo_img = Image.open(BytesIO(response.content))

# Création d'une structure de colonnes pour aligner le titre et l'image
col1, col2, col3 = st.columns([1, 2, 1])

with col2:  # Utilisation de la colonne centrale
    # Affichage du titre 'Team' centré
    st.markdown("<h1 style='text-align: center'>Team</h1>", unsafe_allow_html=True)    
    st.image(logo_img)

# Sélection du pourcentage de précision
precision = st.selectbox('Choisissez le pourcentage de précision :', 
                         ['30%', '40%', '45%', '55%', '60%'])

# Afficher les modèles en fonction de la précision choisie
# Cela dépend de la manière dont vous associez les pourcentages aux modèles
if precision == '30%':
    models = ['Random Forest (36%)', 'Modèle B']  # Exemple
elif precision == '45%':
    models = ['Logistic Regression (45%)', 'Model ']
elif precision == '40%' :
    models = ['Support Vector Machine (41,5%)']  # Exemple
# Ajoutez d'autres conditions pour les autres pourcentages

model_choice = st.selectbox('Choisissez un modèle :', models)

# Zone de texte pour la saisie de la phrase
user_input = st.text_area("Entrez votre texte ici:")

# Chargement et prédiction du modèle
if st.button(f'Prédire le niveau de langue avec {model_choice}'):
    model_url = f"https://github.com/Oglo/Project-DSML/raw/main/Streamlit/{model_choice}.joblib"
    model = load_model(model_url)
    prediction = model.predict([user_input])

    # Dictionnaire pour mapper les chiffres aux niveaux de langue
    niveau_langue = {0: 'A1', 1: 'A2', 2: 'B1', 3: 'B2', 4: 'C1', 5: 'C2'}

    # Affichage du niveau de langue correspondant
    predicted_level = niveau_langue[prediction[0]]
    st.write(f'Niveau de langue prédit: {predicted_level}')
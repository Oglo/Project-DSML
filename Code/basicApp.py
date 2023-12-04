import streamlit as st
import requests
from PIL import Image
from io import BytesIO

# URL de l'image sur GitHub en mode raw
logo_url = "https://raw.githubusercontent.com/Oglo/Project-DSML/main/Code/images/logomigros.png"

# Télécharger l'image depuis l'URL
response = requests.get(logo_url)
logo_img = Image.open(BytesIO(response.content))

# Création d'une structure de colonnes pour aligner le titre et l'image
col1, col2, col3 = st.columns([1, 2, 1])

with col2:  # Utilisation de la colonne centrale
    # Affichage du titre 'Team' centré
    st.markdown("<h1 style='text-align: center'>Team</h1>", unsafe_allow_html=True)

    # Affichage de l'image centrée sous le titre
    st.image(logo_img)

for _ in range(5):  # Ajustez le nombre 5 pour augmenter ou diminuer l'espace
    st.write("")

# Affichage du texte 'Hello World2'
st.write('Hello World2')

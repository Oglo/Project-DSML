import streamlit as st
from PIL import Image
import requests
from io import BytesIO

# URL de l'image sur GitHub en mode raw
logo_url = "https://raw.githubusercontent.com/Oglo/Project-DSML/main/Code/images/logomigros.png"

# Télécharger l'image depuis l'URL
response = requests.get(logo_url)
logo_img = Image.open(BytesIO(response.content))

# Utilisation de colonnes pour aligner le titre et le logo
col1, col2 = st.columns([2, 1])

with col1:
    # Affichage du titre 'Team'
    st.markdown("<h1 style='display: inline-block'>Team</h1>", unsafe_allow_html=True)

with col2:
    # Affichage du logo
    st.image(logo_img, width=100)  # Ajustez la largeur selon vos besoins

# Autres éléments de la page
st.write('Hello World')

import streamlit as st
import requests
from PIL import Image
from io import BytesIO

# URL de l'image sur GitHub en mode raw
logo_url = "https://raw.githubusercontent.com/Oglo/Project-DSML/main/Code/images/logomigros.png"

# Télécharger l'image depuis l'URL
response = requests.get(logo_url)
logo_img = Image.open(BytesIO(response.content))

# Affichage du titre 'Team' centré
st.markdown("<h1 style='text-align: center'>Team</h1>", unsafe_allow_html=True)

# Affichage de l'image centrée sous le titre
st.image(logo_img, width=10, use_column_width=False)  # Ajustez la largeur selon vos besoins

# Autres éléments de la page
st.write('Hello World2')

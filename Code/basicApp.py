import streamlit as st
from PIL import Image
import requests
from io import BytesIO

# URL de l'image sur GitHub en mode raw
logo_url = "https://raw.githubusercontent.com/Oglo/Project-DSML/main/Code/images/logomigros.png"

# Télécharger l'image depuis l'URL
response = requests.get(logo_url)
logo_img = Image.open(BytesIO(response.content))

# Utilisation de HTML pour aligner le titre et le logo
st.markdown("""
    <style>
        .container {
            display: flex;
            align-items: center;
        }
        .logo {
            width: 150px; /* Augmentez la largeur pour un logo plus grand */
            margin-left: 0,005px; /* Réduisez la marge pour déplacer le logo vers la gauche */
        }
    </style>
    <div class="container">
        <h1 style='display: inline-block'>Team</h1>
        <img src="%s" class="logo">
    </div>
""" % logo_url, unsafe_allow_html=True)

# Autres éléments de la page
st.write('Hello World2')

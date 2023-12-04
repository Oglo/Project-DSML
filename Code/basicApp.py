import streamlit as st
from PIL import Image
import requests
from io import BytesIO

# URL de l'image sur GitHub en mode raw
logo_url = "https://raw.githubusercontent.com/Oglo/Project-DSML/main/Code/images/logomigros.png"

# Télécharger l'image depuis l'URL
response = requests.get(logo_url)
logo_img = Image.open(BytesIO(response.content))

# Utilisation de HTML pour centrer le titre et l'image
st.markdown("""
    <style>
        .title {
            text-align: center;
            font-size: 50px; /* Ajustez la taille de police si nécessaire */
        }
        .logo {
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 50%; /* Ajustez la largeur de l'image si nécessaire */
        }
    </style>
    <div class="title">Team</div>
    <img src="%s" class="logo">
""" % logo_url, unsafe_allow_html=True)

# Autres éléments de la page
st.write('Hello World2')

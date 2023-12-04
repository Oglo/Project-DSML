import streamlit as st
import requests
from PIL import Image
from io import BytesIO

# URL de l'image sur GitHub en mode raw
logo_url = "https://raw.githubusercontent.com/Oglo/Project-DSML/main/Code/images/logomigros.png"

# Télécharger l'image depuis l'URL
response = requests.get(logo_url)
logo_img = Image.open(BytesIO(response.content))

# Redimensionner l'image (par exemple, à la moitié de sa taille originale)
logo_img = logo_img.resize((logo_img.width // 2, logo_img.height // 2))

# Affichage du titre 'Team' centré
st.markdown("<h1 style='text-align: center'>Team</h1>", unsafe_allow_html=True)

# Créer des colonnes pour contrôler la largeur de l'image
col1, col2, col3 = st.columns([1,2,1])

with col2:  # Utiliser la colonne centrale pour l'image
    st.image(logo_img)

# Autres éléments de la page
st.write('Hello World2')

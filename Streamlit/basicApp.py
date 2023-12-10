import streamlit as st
import requests
from PIL import Image  # Importer PIL pour ouvrir des images
from io import BytesIO  # Importer BytesIO (corrigez 'import BytesIO')


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
    st.image(logo_img)

for _ in range(5):  
    st.write("")

# Affichage du texte d'accueil
st.write('Welcome to the Migros Team streamlit application! With this platform, you will be able to choose the precision you want in predicting the language level of your text.')

# Espacement
st.write("")

# Création des boutons pour choisir le taux de précision
precision = st.radio("Choisissez le taux de précision :", ('25%', '40%', '50%', '55%', '65%'))

# Affichage de la méthode en fonction du taux de précision sélectionné
if precision == '25%':
    st.write("None")
elif precision == '40%':
    st.write("None")
elif precision == '50%':
    st.write("Logistic regression (45% Accurancy)")
elif precision == '55%':
    st.write("None")
elif precision == '65%':
    st.write("None")

# Espacement
st.write("")

# Création d'un champ de saisie de texte pour que l'utilisateur puisse entrer le nom d'une méthode
user_input = st.text_input("Entrez le nom de la méthode pour obtenir des détails :", "")

# Vérification de la saisie de l'utilisateur et affichage de la réponse
if user_input.lower() == 'Logistic regression':
    texte1 = "Tout le code nécessaire pour run la Logistic Regression"
    st.write(texte1)

    # Bouton pour copier le texte
    st.markdown(f"""
        <textarea id="text_to_copy" style="display: none;">{texte1}</textarea>
        <button onclick="navigator.clipboard.writeText(document.getElementById('text_to_copy').value)">
            Copier le texte
        </button>
    """, unsafe_allow_html=True)

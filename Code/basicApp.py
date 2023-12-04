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

    
    st.image(logo_img)

for _ in range(5):  
    st.write("")

# Affichage du texte 'Hello World2'
st.write('Welcome to the Migros Team streamlit application!With this platform, you will be able to choose the precision you want in predicting the language level of your text.')

# ... [le reste de votre code existant]

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
    st.write("Méthode TF-IF (49% de précision)")
elif precision == '55%':
    st.write("None")
elif precision == '65%':
    st.write("None")



# Espacement
st.write("")

# Création d'un champ de saisie de texte pour que l'utilisateur puisse entrer le nom d'une méthode
user_input = st.text_input("Entrez le nom de la méthode pour obtenir des détails :", "")

# Vérification de la saisie de l'utilisateur et affichage de la réponse
if user_input.lower() == 'méthode 4':
    st.write("ICD vaut 24 points pour égale.")

# Vous pouvez ajouter d'autres conditions pour d'autres méthodes si nécessaire

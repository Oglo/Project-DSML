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

# Affichage du texte d'accueil
st.write('Welcome to the Migros Team streamlit application! With this platform, you will be able to choose the precision you want in predicting the language level of your text.')

# Espacement
st.write("")

# Création des boutons pour choisir le taux de précision
precision = st.radio("Choisissez le taux de précision :", ('30%', '40%', '50%', '55%', '60%'))

# Affichage de la méthode en fonction du taux de précision sélectionné
if precision == '30%':
    st.write("Random Forest (36% Accurancy)")
elif precision == '40%':
    st.write("None")
elif precision == '50%':
    st.write("Logistic regression (45% Accurancy)")
elif precision == '55%':
    st.write("None")
elif precision == '60%':
    st.write("oui")

# Espacement
st.write("")

# Création d'un champ de saisie de texte pour que l'utilisateur puisse entrer le nom d'une méthode
user_input = st.text_input("Entrez le nom de la méthode pour obtenir des détails :", "")

# Vérification de la saisie de l'utilisateur et redirection vers une URL
if user_input.lower() == 'logistic regression':
    url = "https://github.com/Oglo/Project-DSML/blob/main/Code/Methods/1_LogisticRegression.ipynb"  # Remplacez ceci par l'URL de votre choix
    st.markdown(f"[Click here to use Logistic Regression Model]({url})", unsafe_allow_html=True)
if user_input.lower() == 'random forest':
    url = "https://github.com/Oglo/Project-DSML/blob/main/Code/Methods/2_RandomForestClassifier.ipynb"  # Remplacez ceci par l'URL de votre choix
    st.markdown(f"[Click here to use Random Forest Model]({url})", unsafe_allow_html=True)


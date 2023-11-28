import streamlit as st

# Chemin vers votre fichier logo (par exemple, 'logo.png')
logo = "Users/antoi/OneDrive/Bureau/logomigros-removebg-preview.png"

# Utilisation de colonnes pour aligner le logo et le titr
col1, col2 = st.beta_columns([1, 10])

with col1:
    st.image(logo, width=50)  # Vous pouvez ajuster la largeur selon vos besoins

with col2:
    st.title('Team Migros')

st.write('Hello World')

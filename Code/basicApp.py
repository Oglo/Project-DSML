import streamlit as st

# URL de l'image sur GitHub
logo_url = "https://raw.githubusercontent.com/user/repo/branch/path/to/image.png"

# Utilisation de colonnes pour aligner le logo et le titre
col1, col2 = st.columns([1, 10])

with col1:
    st.image(logo_url, width=50)  # Ajustez la largeur selon vos besoins

with col2:
    st.title('Team Migros')

st.write('Hello World')

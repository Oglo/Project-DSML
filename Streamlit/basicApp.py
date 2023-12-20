import streamlit as st
import joblib
import requests
import tempfile
import spacy
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from io import BytesIO
import torch
from PIL import Image
import gdown
from youtube_transcript_api import YouTubeTranscriptApi
nlp = spacy.load('fr_core_news_sm')

def download_youtube_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = transcript_list.find_transcript(['fr'])
        transcript_text = transcript.fetch()
        full_transcript = "\n".join([t['text'] for t in transcript_text])
        return full_transcript
    except Exception as e:
        return f"Erreur lors de la récupération des sous-titres: {e}"

def extract_video_id_from_url(url):
    import re
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    return None






def encode_sentences(tokenizer, sentences, max_length):
    input_ids = []
    attention_masks = []

    for sentence in sentences:
        encoded_dict = tokenizer.encode_plus(
            text=sentence,              # Phrase à encoder.
            add_special_tokens=True,    # Ajoutez '[CLS]' et '[SEP]'
            max_length=max_length,      # Longueur maximale de la séquence (coupe et pad si nécessaire).
            pad_to_max_length=True,     # Pad & Truncate toutes les phrases.
            return_attention_mask=True, # Construire les masques d'attention.
            return_tensors='pt',        # Renvoie les tensors PyTorch.
        )

        # Ajoutez l'encodage résultant à la liste.
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    # Convertissez les listes en tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return input_ids, attention_masks






def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zàâçéèêëîïôûùüÿñæœ]', ' ', text)
    words = word_tokenize(text, language='french')
    words = [word for word in words if word not in stopwords.words('french')]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)



def convert_to_label(prediction):
    # Supposons que prediction est un entier correspondant à une classe
    difficulty_mapping = {0: 'A1', 1: 'A2', 2: 'B1', 3: 'B2', 4: 'C1', 5: 'C2'}
    return difficulty_mapping.get(prediction[0], "Inconnu")



def extract_features(text):
    doc = nlp(text)
    num_sentences = len(list(doc.sents))
    num_tokens = len(doc)
    avg_sentence_length = num_tokens / num_sentences if num_sentences > 0 else 0
    lexical_diversity = len(set(token.text.lower() for token in doc)) / num_tokens if num_tokens > 0 else 0
    return [avg_sentence_length, lexical_diversity]




# Fonction pour charger un modèle depuis GitHub
def load_model_from_github(url):
    response = requests.get(url)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as tmp_file:
        tmp_file.write(response.content)
        tmp_file.flush()
        loaded_object = joblib.load(tmp_file.name)
    return loaded_object








# Interface utilisateur
logo_url = "https://raw.githubusercontent.com/Oglo/Project-DSML/main/Code/images/logomigros.png"
logo_yt = "https://raw.githubusercontent.com/Oglo/Project-DSML/main/Code/images/logoyt.png"
# Télécharger l'image depuis l'URL
response = requests.get(logo_url)
logo_img = Image.open(BytesIO(response.content))
response2 = requests.get(logo_yt)
image_yt = Image.open(BytesIO(response2.content))

# Création d'une structure de colonnes pour aligner le titre et l'image
col1, col2, col3 = st.columns([1, 2, 1])

with col2:  # Utilisation de la colonne centrale
    st.markdown("<h1 style='text-align: center'>Team</h1>", unsafe_allow_html=True)    
    st.image(logo_img)







def main():
    
    model_choice = None
    # Sélection de la précision du modèle
    precision = st.selectbox("Choisissez le pourcentage de précision:", ["30%", "40%", "50%", "55%"])
    model = None
    vectorizer = None
    if precision == "30%":
        model_choice = st.selectbox("Choisissez le modèle:", ["Random Forest"])
        
    elif precision == "40%":
        model_choice = st.selectbox("Choisissez le modèle:", ["Logistic Regression", "Spacy"])

    elif precision == "50%":
        model_choice = st.selectbox("Choisissez le modèle:", ["Vector"])
        
            
    sentence = st.text_area("Entrez une phrase:")

    # Bouton de prédiction
    if st.button("Prédire"):

        if  model_choice == "Random Forest":
                model = load_model_from_github("https://github.com/Oglo/Project-DSML/raw/main/Streamlit/Random Forest.joblib")
                vectorizer = load_model_from_github("https://github.com/Oglo/Project-DSML/raw/main/Streamlit/vectorizer.joblib")
                transformed_sentence = vectorizer.transform([sentence])
                prediction = model.predict(transformed_sentence)
                difficulty_label = convert_to_label(prediction)
                st.write(f"Prédiction de la difficulté: {prediction}")

        elif model_choice == "Logistic Regression":
                model = load_model_from_github("https://github.com/Oglo/Project-DSML/raw/main/Streamlit/Logistic Regression (45%).joblib")
                prediction = model.predict([sentence])  # Assurez-vous que cela correspond au format attendu par le modèle
                difficulty_label = convert_to_label(prediction)
                st.write(f"Prédiction de la difficulté: {difficulty_label}")

        elif model_choice == "Spacy":
                model = load_model_from_github("https://github.com/Oglo/Project-DSML/raw/main/Streamlit/Spacy (32%).joblib")
                features = extract_features(sentence)
                prediction = model.predict([features]) # Remplacez 'features' par les caractéristiques extraites
                difficulty_label = convert_to_label(prediction)
                st.write(f"Prédiction de la difficulté: {difficulty_label}")

        elif  model_choice == "Vector":
                model = load_model_from_github("https://github.com/Oglo/Project-DSML/raw/main/Streamlit/Vector.joblib")
                vectorizer = load_model_from_github("https://github.com/Oglo/Project-DSML/raw/main/Streamlit/vectorizer.joblib")
                processed_sentence = preprocess_text(sentence)
                transformed_sentence = vectorizer.transform([processed_sentence])
                prediction = model.predict(transformed_sentence)
                difficulty_label = convert_to_label(prediction)
                st.write(f"Prédiction de la difficulté: {prediction}")







    st.markdown("### Prédiction du Niveau de Langue des Sous-titres YouTube")
    youtube_url = st.text_input("Entrez l'URL d'une vidéo YouTube ici :")

    # Bouton de prédiction pour les sous-titres
    if st.button("Prédire le Niveau de Langue des Sous-titres"):
        if youtube_url:
            video_id = extract_video_id_from_url(youtube_url)
            if video_id:
                subtitles = download_youtube_transcript(video_id)
                st.text_area("Sous-titres:", value=subtitles, height=150)
                processed_subtitles = preprocess_text(subtitles)

                # Ici, utilisez le modèle choisi pour la prédiction
                if model_choice == "Logistic Regression":
                     model = load_model_from_github("https://github.com/Oglo/Project-DSML/raw/main/Streamlit/Logistic Regression (45%).joblib")
                     prediction = model.predict([processed_subtitles])
                     difficulty_label = convert_to_label(prediction)
                     st.write(f"Prédiction de la difficulté: {difficulty_label}")
                    # Assurez-vous que le modèle Logistic Regression est chargé ici
                    # ... Logique de prédiction pour Logistic Regression ...
                # Ajoutez des conditions pour d'autres modèles si nécessaire
        else:
            st.error("Veuillez entrer une URL YouTube.")    


if __name__ == "__main__":
    main()
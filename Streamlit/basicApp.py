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
from transformers import FlaubertTokenizer, FlaubertForSequenceClassification
from youtube_transcript_api import YouTubeTranscriptApi
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf


nlp = spacy.load('fr_core_news_sm')


def load_rnn_model_and_tokenizer():
    model_url = "https://github.com/Oglo/Project-DSML/raw/main/Streamlit/mon_modele.h5"
    tokenizer_url = "https://github.com/Oglo/Project-DSML/raw/main/Streamlit/tokenizer.joblib"
    
    rnn_model = load_keras_model_from_github(model_url)
    rnn_tokenizer = load_model_from_github(tokenizer_url)

    return rnn_model, rnn_tokenizer


def predict_with_rnn(sentence, tokenizer, model):
    sequence = tokenizer.texts_to_sequences([sentence])
    padded_sequence = pad_sequences(sequence, maxlen=512)
    prediction = model.predict(padded_sequence)
    predicted_label = np.argmax(prediction, axis=-1)
    return predicted_label



def load_flaubert_model(gdrive_url):
    file_id = gdrive_url.split('=')[-1] 
    destination = 'FlauBERT_model.pth'
    
    
    gdown.download(url=f"https://drive.google.com/uc?id={file_id}", output=destination, quiet=False)
    
    model = FlaubertForSequenceClassification.from_pretrained('flaubert/flaubert_base_cased', num_labels=6)
    model.load_state_dict(torch.load(destination, map_location=torch.device('cpu')))
    return model


def predict_with_flaubert(text, tokenizer, model, device):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = outputs.logits.softmax(1)
    prediction = probs.argmax().item()
    return prediction


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
            text=sentence,              
            add_special_tokens=True,    
            max_length=max_length,      
            pad_to_max_length=True,     
            return_attention_mask=True, 
            return_tensors='pt',        
        )

        
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    
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
    
    difficulty_mapping = {0: 'A1', 1: 'A2', 2: 'B1', 3: 'B2', 4: 'C1', 5: 'C2'}
    return difficulty_mapping.get(prediction[0], "Inconnu")


def convert_to_label_invers(prediction):
    
    difficulty_mapping_invers =  {'A1': 0, 'A2': 1, 'B1': 2, 'B2': 3, 'C1': 4, 'C2': 5}
    
    for key, value in difficulty_mapping_invers.items():
        if value == prediction:
            return key
    return "Inconnu"


def extract_features(text):
    doc = nlp(text)
    num_sentences = len(list(doc.sents))
    num_tokens = len(doc)
    avg_sentence_length = num_tokens / num_sentences if num_sentences > 0 else 0
    lexical_diversity = len(set(token.text.lower() for token in doc)) / num_tokens if num_tokens > 0 else 0
    return [avg_sentence_length, lexical_diversity]


def load_model_from_github(url):
    response = requests.get(url)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as tmp_file:
        tmp_file.write(response.content)
        tmp_file.flush()
        loaded_object = joblib.load(tmp_file.name)
    return loaded_object


def load_keras_model_from_github(url):
    response = requests.get(url)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
        tmp_file.write(response.content)
        tmp_file.flush()
        loaded_model = tf.keras.models.load_model(tmp_file.name)
    return loaded_model












logo_url = "https://raw.githubusercontent.com/Oglo/Project-DSML/main/Code/images/logomigros.png"
logo_yt = "https://raw.githubusercontent.com/Oglo/Project-DSML/main/Code/images/logoyt.png"

response = requests.get(logo_url)
logo_img = Image.open(BytesIO(response.content))
response2 = requests.get(logo_yt)
image_yt = Image.open(BytesIO(response2.content))


col1, col2, col3 = st.columns([1, 2, 1])

with col2:  
    st.markdown("<h1 style='text-align: center'>Team</h1>", unsafe_allow_html=True)    
    st.image(logo_img)







def main():
    
    model_choice = None
    # Sélection de la précision du modèle
    precision = st.selectbox("Choose your accurancy:", ["30%", "40%", "50%", "55%"])
    model = None
    vectorizer = None
    if precision == "30%":
        model_choice = st.selectbox("Choose your model:", ["Random Forest 35%", 'Spacy 33%'])
        
    elif precision == "40%":
        model_choice = st.selectbox("Choose your model:", ["Logistic Regression 45%", 'Reccurent Neural Network 44,7%', 'Support Vector Machine 42%'])

    elif precision == "50%":
        model_choice = st.selectbox("Choose your model:", ["None"])

    elif precision == "55%":
        model_choice = st.selectbox("Choose your model:", ['FlauBERT 56,5%'])
        
            
    sentence = st.text_area("Write your sentence here:")

    # Bouton de prédiction
    if st.button(f"Predict with {model_choice}"):

        if  model_choice == "Random Forest 35%":
                model = load_model_from_github("https://github.com/Oglo/Project-DSML/raw/main/Streamlit/Random_Forest.joblib")
                vectorizer = load_model_from_github("https://github.com/Oglo/Project-DSML/raw/main/Streamlit/vectorizer.joblib")
                transformed_sentence = vectorizer.transform([sentence])
                prediction = model.predict(transformed_sentence)
                difficulty_label = convert_to_label(prediction)
                st.write(f"Difficulty level: {prediction}")

        elif model_choice == "Logistic Regression 45%":
                model = load_model_from_github("https://github.com/Oglo/Project-DSML/raw/main/Streamlit/Logistic_Regression.joblib")
                prediction = model.predict([sentence])  # Assurez-vous que cela correspond au format attendu par le modèle
                difficulty_label = convert_to_label(prediction)
                st.write(f"Difficulty level: {difficulty_label}")

        elif model_choice == "Spacy 33%":
                model = load_model_from_github("https://github.com/Oglo/Project-DSML/raw/main/Streamlit/Spacy.joblib")
                features = extract_features(sentence)
                prediction = model.predict([features]) # Remplacez 'features' par les caractéristiques extraites
                difficulty_label = convert_to_label(prediction)
                st.write(f"Difficulty level: {difficulty_label}")

        elif  model_choice == "Support Vector Machine 42%":
                model = load_model_from_github("https://github.com/Oglo/Project-DSML/raw/main/Streamlit/Vector.joblib")
                vectorizer = load_model_from_github("https://github.com/Oglo/Project-DSML/raw/main/Streamlit/vectorizer.joblib")
                processed_sentence = preprocess_text(sentence)
                transformed_sentence = vectorizer.transform([processed_sentence])
                prediction = model.predict(transformed_sentence)
                difficulty_label = convert_to_label(prediction)
                st.write(f"Difficulty level: {prediction}")

        elif model_choice == "FlauBERT 56,5%":
            st.markdown("This model may take a while.")
            gdrive_url = "https://drive.google.com/uc?id=1Sa6u3SUHSVylnNuFoxh-ibQ1mnXH48zx"
            model = load_flaubert_model(gdrive_url)
            tokenizer = FlaubertTokenizer.from_pretrained('flaubert/flaubert_base_cased')
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            prediction_numeric = predict_with_flaubert(sentence, tokenizer, model, device)
            difficulty_mapping_invers = convert_to_label_invers(prediction_numeric)
            st.write(f"Difficulty level: {difficulty_mapping_invers}")
            
        elif model_choice == "Reccurent Neural Network 44,7%":
            rnn_model, rnn_tokenizer = load_rnn_model_and_tokenizer()
            prediction = predict_with_rnn(sentence, rnn_tokenizer, rnn_model)
            difficulty_label = convert_to_label(prediction)
            st.write(f"Difficulty level: {difficulty_label}")    





    # Création de colonnes pour l'image et le texte
    col_img, col_text = st.columns([1, 2])

    with col_img:
    # Affichage de l'image avec une largeur spécifique
        st.image(image_yt, width=150)  # Ajustez la valeur de 'width' selon vos besoins

    with col_text:
    # Affichage du texte à droite de l'image
        st.markdown("<h1 style='text-align: left'>YouTube</h1>", unsafe_allow_html=True)


    st.markdown("Now, let's predict subtiles difficulty")
    youtube_url = st.text_input("Past Youtube URL here :")

    
            
    
    if st.button(f"Predict subtiles difficulty with {model_choice}"):
        if youtube_url:
            video_id = extract_video_id_from_url(youtube_url)
            if video_id:
                subtitles = download_youtube_transcript(video_id)
                st.text_area("Sous-titres:", value=subtitles, height=150)
                processed_subtitles = preprocess_text(subtitles)

                # Ici, utilisez le modèle choisi pour la prédiction
                if model_choice == "Logistic Regression 45%":
                     model = load_model_from_github("https://github.com/Oglo/Project-DSML/raw/main/Streamlit/Logistic_Regression.joblib")
                     prediction = model.predict([processed_subtitles])
                     difficulty_label = convert_to_label(prediction)
                     st.write(f"Difficulty level: {difficulty_label}")
                    
                elif  model_choice == "Support Vector Machine 42%":
                    model = load_model_from_github("https://github.com/Oglo/Project-DSML/raw/main/Streamlit/Vector.joblib")
                    vectorizer = load_model_from_github("https://github.com/Oglo/Project-DSML/raw/main/Streamlit/vectorizer.joblib")
                    processed_sentence = preprocess_text(subtitles)
                    transformed_sentence = vectorizer.transform([processed_subtitles])
                    prediction = model.predict(transformed_sentence)
                    difficulty_label = convert_to_label(prediction)
                    st.write(f"Difficulty level: {prediction}")

                elif model_choice == "Spacy 33%":
                    model = load_model_from_github("https://github.com/Oglo/Project-DSML/raw/main/Streamlit/Spacy.joblib")
                    features = extract_features(subtitles)
                    prediction = model.predict([features]) # Remplacez 'features' par les caractéristiques extraites
                    difficulty_label = convert_to_label(prediction)
                    st.write(f"Difficulty level: {difficulty_label}")  

                elif  model_choice == "Random Forest 35%":
                    model = load_model_from_github("https://github.com/Oglo/Project-DSML/raw/main/Streamlit/Random_Forest.joblib")
                    vectorizer = load_model_from_github("https://github.com/Oglo/Project-DSML/raw/main/Streamlit/vectorizer.joblib")
                    transformed_sentence = vectorizer.transform([processed_subtitles])
                    prediction = model.predict(transformed_sentence)
                    difficulty_label = convert_to_label(prediction)
                    st.write(f"Difficulty level: {prediction}")  

                elif model_choice == "FlauBERT 56,5%":

                    gdrive_url = "https://drive.google.com/uc?id=1Sa6u3SUHSVylnNuFoxh-ibQ1mnXH48zx"
                    model = load_flaubert_model(gdrive_url)
                    tokenizer = FlaubertTokenizer.from_pretrained('flaubert/flaubert_base_cased')
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    model.to(device)
                    prediction_numeric = predict_with_flaubert(subtitles, tokenizer, model, device)
                    difficulty_mapping_invers = convert_to_label_invers(prediction_numeric)
                    st.write(f"Difficulty level: {difficulty_mapping_invers}")

                elif model_choice == "Reccurent Neural Network 44,7%":
                    rnn_model, rnn_tokenizer = load_rnn_model_and_tokenizer()
                    prediction = predict_with_rnn(subtitles, rnn_tokenizer, rnn_model)
                    difficulty_label = convert_to_label(prediction)
                    st.write(f"Difficulty level: {difficulty_label}")

        else:
            st.error("Please past another URL.")   

      # Utilisation de la colonne centrale
       
         


if __name__ == "__main__":
    main()


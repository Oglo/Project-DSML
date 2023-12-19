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
from transformers import BertTokenizer, BertForSequenceClassification
from PIL import Image
import gdown
url = 'https://drive.google.com/uc?id=1ng8ugkZtxQlfkohAcgLLUSVCbkSsg_F8'
output = 'model_bert.pth'
gdown.download(url, output, quiet=False)
nlp = spacy.load('fr_core_news_sm')


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





def predict_with_bert(model, tokenizer, sentence):
    processed_sentence = preprocess_text(sentence)
    input_ids, attention_masks = encode_sentences(tokenizer, [processed_sentence], max_length=256)

    with torch.no_grad():
        outputs = model(input_ids, token_type_ids=None, attention_mask=attention_masks)
        predictions = torch.argmax(outputs.logits, dim=1)
    
    return predictions[0].item()







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
    

    # Sélection de la précision du modèle
    precision = st.selectbox("Choisissez le pourcentage de précision:", ["30%", "40%", "50%", "55%"])
    model = None
    vectorizer = None
    if precision == "30%":
        model_choice = st.selectbox("Choisissez le modèle:", ["Random Forest"])
        if model_choice == "Random Forest":
                model = load_model_from_github("https://github.com/Oglo/Project-DSML/raw/main/Streamlit/Random Forest.joblib")
                vectorizer = load_model_from_github("https://github.com/Oglo/Project-DSML/raw/main/Streamlit/vectorizer.joblib")
    
    elif precision == "40%":
        model_choice = st.selectbox("Choisissez le modèle:", ["Logistic Regression", "Spacy"])

    elif precision == "50%":
        model_choice = st.selectbox("Choisissez le modèle:", ["Vector",'Bert'])
        if model_choice == "Vector":
                model = load_model_from_github("https://github.com/Oglo/Project-DSML/raw/main/Streamlit/Vector.joblib")
                vectorizer = load_model_from_github("https://github.com/Oglo/Project-DSML/raw/main/Streamlit/vectorizer.joblib")
        elif  model_choice == "Bert":
                tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
                model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=6)
            
    sentence = st.text_area("Entrez une phrase:")

    # Bouton de prédiction
    if st.button("Prédire"):

        if  model_choice == "Random Forest":
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
                processed_sentence = preprocess_text(sentence)
                transformed_sentence = vectorizer.transform([processed_sentence])
                prediction = model.predict(transformed_sentence)
                difficulty_label = convert_to_label(prediction)
                st.write(f"Prédiction de la difficulté: {prediction}")

        elif model_choice == "Bert":
                prediction = predict_with_bert(sentence)
                difficulty_label = convert_to_label(prediction)
                st.write(f"Prédiction de la difficulté: {difficulty_label}")
        

        # Afficher les résultats de la prédiction

if __name__ == "__main__":
    main()

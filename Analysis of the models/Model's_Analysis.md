# Confusion Matrix Analysis

Logistic Regression, Random Forest, SVM: These traditional machine learning models are generally good at capturing linear relationships. Their confusion matrices will show how well they categorize each class. A balanced matrix (with higher values along the diagonal) indicates good performance. However, if off-diagonal numbers are high, it points to misclassifications.

**Spacy:** Being an NLP-focused tool, Spacy's matrix should ideally show better handling of language nuances compared to more general models. Misclassifications here might suggest challenges in understanding context or complex language structures.

**RNN:** As a deep learning model, it's proficient in capturing sequential data. Its confusion matrix should ideally reflect a good understanding of context in text data. High off-diagonal values could indicate issues with understanding long-term dependencies or overfitting.

**FlauBERT:** Specifically tailored for French, its confusion matrix should ideally show fewer misclassifications, reflecting its strength in understanding the nuances of French text.

**Logistic Regresssion**
<p align="center">
    <img src="https://github.com/Oglo/Project-DSML/blob/main/Code/images/Confusion_Logistic_regression.jpeg" alt="First steps of Trello" width="300"/>
</p>

**Random Forest Classifier**
<p align="center">
    <img src="https://github.com/Oglo/Project-DSML/blob/main/Code/images/Confusion_RandomForest.jpeg" alt="First steps of Trello" width="300"/>
</p>

**RNN**
<p align="center">
    <img src="https://github.com/Oglo/Project-DSML/blob/main/Code/images/Confusion_RNN.png" alt="First steps of Trello" width="300"/>
</p>

**Spacy**
<p align="center">
    <img src="https://github.com/Oglo/Project-DSML/blob/main/Code/images/Confusion_Spacy.jpeg" alt="First steps of Trello" width="300"/>
</p>

**SVM**
<p align="center">
    <img src="https://github.com/Oglo/Project-DSML/blob/main/Code/images/Confusion_SVM.jpeg" alt="First steps of Trello" width="300"/>
</p>

**FlauBERT**
<p align="center">
    <img src="https://github.com/Oglo/Project-DSML/blob/main/Code/images/Confusion_FlauBERT.jpeg" alt="First steps of Trello" width="300"/>
</p>


# Analysis of Table Metrics

**Accuracy:** Measures the overall correctness of the model. Higher accuracy across all models is good, but it doesn't account for class imbalances.

**Precision:** Indicates the proportion of positive identifications that were actually correct. Models with higher precision are better at minimizing false positives.

**F1-Score:** A balance between precision and recall. It's particularly useful if you have an uneven class distribution. Higher F1-scores across your models indicate a good balance between precision and recall.

**Recall:** Measures the proportion of actual positives that were identified correctly. High recall indicates the model is good at minimizing false negatives.

<p align="center">
    <img src="https://github.com/Oglo/Project-DSML/blob/main/Code/images/Table.png" alt="First steps of Trello" width="300"/>
</p>

In summary, comparing these metrics across different models gave us insight into their strengths and weaknesses. For instance, traditional models like Logistic Regression might show high accuracy but lower recall compared to deep learning models like RNNs or FlauBERT, which can better capture complex patterns in text data.
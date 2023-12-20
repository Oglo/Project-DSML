<h1>Project-DSML - TEAM <span style="color: orange;">MIGROS</span></h1>



Hello everyone, we're Antoine Oglobleff and Manuel Solazzo and we're the Migros team for this Project.

Here you'll find all the files we've worked on to enable us to have a program that can predict the language level of a French text. We'll also put in all the thoughts that led us to use the right method.

You can find a YouTube video that explain the project below by clicking on the image :

[![Miniature de la vidéo](https://github.com/Oglo/Project-DSML/blob/main/Code/images/YTVideoImage.jpeg)](https://www.youtube.com/watch?v=o2FGTH8IWBA)

 

We are pleased to present this detailed guide to navigating our GitHub repository.

-------

Analysis of the Model Folder: 

This primary folder contains a thorough analysis of our various models.

Erroneous Predictions File:

- Here, we examine models that did not meet our performance criteria, highlighting the reasons for their insufficient accuracy.


The Best Model File: 

- This document details the model that outperformed others in terms of prediction.

Model's Dashboard File:

- A comprehensive overview of key metrics such as F1-score, recall, accuracy, precision, and the confusion matrix for our most significant models.

---

Code Folder: 

- Explore the source code of all our models, complete with explanations of their operation. We recommend reviewing the 'DISCLAIMERS' file for additional information. This folder also includes a sub-folder containing all images used on our GitHub and Streamlit.

----

Data Folder: 

- Here, you'll find all the data you provided to us, along with a 'training_data_2' CSV file representing a duplication of the training data.

----

Our Progress Folder: 

- This folder offers an in-depth view of our team's methods and working processes, highlighting efficiency and collaboration within our group.

----

Streamlit Folder:


- Contains the code for our application 'basicApp.py', integrating all the aforementioned models. It serves as the platform for running our application.

----

We hope this guide will make it easier for you to navigate and understand our work on GitHub. Please do not hesitate to contact us for any further questions or clarifications.


You can contact us by e-mail here :

<img src="https://raw.githubusercontent.com/Oglo/Project-DSML/main/Code/images/logomail.png" alt="Email Logo" width="35" height="20">      antoine.oglobleff@unil.ch

<img src="https://raw.githubusercontent.com/Oglo/Project-DSML/main/Code/images/logomail.png" alt="Email Logo" width="35" height="20">      manuel.solazzo@unil.ch





Then, a detailed guide to enhance your experience with our Streamlit application.

First, launch the basicApp.py file on your local system. This will redirect you to the main page of our application. On this interface, you will have the opportunity to select a specific percentage, which will be used to predict the difficulty level of the desired content. Following this selection, you can choose from a variety of available models the one that best suits your needs.

After selecting your model, a text input area will allow you to enter any text in French. By clicking on the 'Predict' button, the previously selected model will analyze the text and provide an assessment of its difficulty level.

Additionally, our application offers a separate feature for YouTube videos. You will have a second text box where you can paste the URL of a YouTube video. Pressing 'Predict Subtitles' will cause the program to extract and display the entirety of the subtitles from the selected video and assess their difficulty using the model you have chosen beforehand.

We hope this guide will help you navigate our application easily and make the most of its features. 

By the way, if you choose the model FlauBERT, the program may take some time to run because it has to download the model via Google Drive and put it in Github before it can be used.
We used this process because, as the model is over 100MB, we couldn't commit it directly to our GitHub.

(So, if you decide to try FlauBERT on our streamlit Application, don't commit or push afterwards)

Have fun !

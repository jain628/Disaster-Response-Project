- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- DisasterResponse.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md

Project Description

Here, a model is build to classify messages that are sent during disasters. There are 36 pre-defined categories, and examples of these categories include Aid Related, Medical Help, Search And Rescue, etc. By classifying these messages, we can allow these messages to be sent to the appropriate disaster relief agency. This project will involve the building of a basic ETL and Machine Learning pipeline to facilitate the task. This is also a multi-label classification task, since a message can belong to one or more categories.Using the web app an emergency worker can input a new message and get classification results in several categories so to have an idea what kind of help is needed: "water", "shelter", "food", etc.
It can be viewed in web_app_screenshot.png

The web app also displays visualizations of the data.
visualizations1.png, visualizations2.png, visualizations3.png gives the feel of data.

Installation
Must runing with Python 3 with libraries of numpy, pandas, sqlalchemy, re, NLTK, pickle, Sklearn, plotly and flask libraries.

File Descriptions
1. app folder : includes the templates folder and "run.py" for the web application
2. data folder : contains "DisasterResponse.db", "disaster_categories.csv", "disaster_messages.csv" and "process_data.py" for data cleaning and transfering.
3. models folder : includes "classifier.pkl" and "train_classifier.py" for the Machine Learning model.
4. README file
5. Preparation folder : contains 2 different files, which were used during the development of project (Please note: this folder is not necessary for this project to run.)

Licensing & Acknowledgement
This app was completed as part of the Udacity Data Scientist Nanodegree. Code templates and data were provided by them. Special thanks to udacity for this training. Feel free to utilize the contents of this while citing me and/or udacity accordingly.

NOTE: Preparation folder is not necessary for this project to run.

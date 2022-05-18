- app<br>
| - template<br>
| |- master.html  # main page of web app<br>
| |- go.html  # classification result page of web app<br>
|- run.py  # Flask file that runs app<br>

- data<br>
|- disaster_categories.csv  # data to process <br>
|- disaster_messages.csv  # data to process<br>
|- process_data.py<br>
|- DisasterResponse.db   # database to save clean data to<br>

- models<br>
|- train_classifier.py<br>
|- classifier.pkl  # saved model <br>

- README.md<br>

###Project Description

Here, a model is build to classify messages that are sent during disasters. There are 36 pre-defined categories, and examples of these categories include Aid Related, Medical Help, Search And Rescue, etc. By classifying these messages, we can allow these messages to be sent to the appropriate disaster relief agency. This project will involve the building of a basic ETL and Machine Learning pipeline to facilitate the task. This is also a multi-label classification task, since a message can belong to one or more categories.Using the web app an emergency worker can input a new message and get classification results in several categories so to have an idea what kind of help is needed: "water", "shelter", "food", etc.
It can be viewed in web_app_screenshot.png<br>

The web app also displays visualizations of the data.
visualizations1.png, visualizations2.png, visualizations3.png gives the feel of data.<br>

###Installation
Must runing with Python 3 with libraries of numpy, pandas, sqlalchemy, re, NLTK, pickle, Sklearn, plotly and flask libraries.<br>

###File Descriptions
1. app folder : includes the templates folder and "run.py" for the web application<br>
2. data folder : contains "DisasterResponse.db", "disaster_categories.csv", "disaster_messages.csv" and "process_data.py" for data cleaning and transfering.
3. models folder : includes "classifier.pkl" and "train_classifier.py" for the Machine Learning model.<br>
4. README file<br>
5. Preparation folder : contains 2 different files, which were used during the development of project <br>(Please note: this folder is not necessary for this project to run.)<br>

###Licensing & Acknowledgement
This app was completed as part of the Udacity Data Scientist Nanodegree. Code templates and data were provided by them. Special thanks to udacity for this training. Feel free to utilize the contents of this while citing me and/or udacity accordingly.<br>

###NOTE: Preparation folder is not necessary for this project to run.

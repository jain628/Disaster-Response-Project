### Workflow
- app<br>
&emsp;| - template<br>
&emsp;&emsp;|- master.html  # main page of web app<br>
&emsp;&emsp;|- go.html  # classification result page of web app<br>
&emsp;| - run.py  # Flask file that runs app<br>

- data<br>
&emsp;| - disaster_categories.csv  # data to process <br>
&emsp;| - disaster_messages.csv  # data to process<br>
&emsp;| - process_data.py<br>
&emsp;| - DisasterResponse.db   # database to save clean data to<br>

- models<br>
&emsp;| - train_classifier.py<br>
&emsp;| - classifier.pkl  # saved model <br>

- README.md<br>

### Project Description

Here, a model is build to classify messages that are sent during disasters. There are 36 pre-defined categories, and examples of these categories include Aid Related, Medical Help, Search And Rescue, etc. By classifying these messages, we can allow these messages to be sent to the appropriate disaster relief agency. This project will involve the building of a basic ETL and Machine Learning pipeline to facilitate the task. This is also a multi-label classification task, since a message can belong to one or more categories.Using the web app an emergency worker can input a new message and get classification results in several categories so to have an idea what kind of help is needed: "water", "shelter", "food", etc.
It can be viewed in web_app.png<br>

The web app also displays visualizations of the data.
visualizations1.png, visualizations2.png, visualizations3.png gives the feel of data.<br>
### Visualization 1
<img src ="https://github.com/jain628/Disaster-Response-Project/blob/main/visualization1.jpg"><br>
### Visualization 2
<img src ="https://github.com/jain628/Disaster-Response-Project/blob/main/visualization2.jpg"><br>
### Visualization 3
<img src ="https://github.com/jain628/Disaster-Response-Project/blob/main/visualization3.jpg"><br>
### web app visualization
<img src ="https://github.com/jain628/Disaster-Response-Project/blob/main/web_app.png"><br>

### Installation
Must runing with Python 3 with libraries of numpy, pandas, sqlalchemy, re, NLTK, pickle, Sklearn, plotly and flask libraries.<br>

### File Descriptions
1. app folder : includes the templates folder and "run.py" for the web application<br>
2. data folder : contains "DisasterResponse.db", "disaster_categories.csv", "disaster_messages.csv" and "process_data.py" for data cleaning and transfering.
3. models folder : includes "classifier.pkl" and "train_classifier.py" for the Machine Learning model.<br>
4. README file<br>
5. Preparation folder : contains 2 different files, which were used during the development of project <br>(Please note: this folder is not necessary for this project to run.)<br>

### Instructions
Run the following commands in the project's root directory to set up your database and model.<br>
1.&ensp;To run ETL pipeline that cleans data and stores in database:<br> &emsp;>>"python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db"<br>
2.&ensp;To run ML pipeline that trains classifier and saves:<br> &emsp;>>"python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl"<br>
3.&ensp;Run the following command in the app's directory to run your web app:<br>&emsp;>>"python run.py"<br>
4.&ensp;Go to http://0.0.0.0:3001/<br>

### Licensing & Acknowledgement
This app was completed as part of the Udacity Data Scientist Nanodegree. Code templates and data were provided by them. Special thanks to udacity for this training. Feel free to utilize the contents of this while citing me and/or udacity accordingly.<br>

NOTE: Preparation folder is not necessary for this project to run.

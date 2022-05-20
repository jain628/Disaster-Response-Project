#importing libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
import pickle
import re
import nltk
nltk.download('punkt')
nltk.download('wordnet')

def load_data(database_filepath):
    """
    Returns two python dataframes and a column after reading DisasterResponse.db from ETL pipeline
    
    Parameters:
        database_filepath (str): a string value for database_filepath
        
    Returns:
        X (python dataframe):a dataframe with messages
        y (python dataframe): a datframe with all the 36 categories having binary values.
        category_names(series): a column with category names
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterResponse',engine)
    df.drop(df[df['genre'] == 0].index, inplace = True)
    X = df.message
    y = (df[df.columns[4:]]).astype(int)
    
    category_names = y.columns
    return X, y, category_names 
    

def tokenize(text):
    """
    Returns cleaned and lemmatized tokens using regex,WordNetLemmatizer()
    
    Parameters:
        text(str): a string value (from X.message)
    
    Returns:
        clean_tokens (str):a string after tokeninzing,cleaning,lemmatizing
    """
    url_regex ='http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex,text)
    for url in detected_urls:
        text = text.replace(url,"urlplaceholder")
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens=[]
    for token in tokens:
        clean_tok = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    """
    Returns a model by implementing CountVectorizer,TfIdfTransformer,MultiOutputClassifier and GridSearchCV .
    
    Parameters:
        None
    
    Returns:
        model : returns GridSearchCV estimator implemented using pipeline.
    """
    pipeline = Pipeline([
    ('vect',CountVectorizer(tokenizer=tokenize)),
    ('tfidf',TfidfTransformer()),
    ('clf',MultiOutputClassifier(RandomForestClassifier()))
    ])
       # parameters set to this due to reduce the size of pkl file, which were too large (600MB) for uploading to github with my previous parameters.
    parameters = {
                    'clf__estimator__n_estimators': [10],
                    'clf__estimator__min_samples_split': [10]
              }
    model = GridSearchCV(pipeline, param_grid=parameters, n_jobs=2, verbose=4)
    return model
    


def evaluate_model(model, X_test, y_test, category_names):
    """
    Evaluates model by predicting over test data
    
    Parameters:
        model(estimator):an estimator returned from build_model() function
        X_text(python dataframe): a dataframe for train data
        y_test(python dataframe):a dataframe for test data
        category_names(python column): column with category names
    
    Returns:
        None
    """
    y_pred = model.predict(X_test)


def save_model(model, model_filepath):
    """
    Saves the model by creating a pickle file named 'classifier.pkl'
    
    Parameters:
        model : model returned by build_model() function
        model_filepath(str): a string value for the model path
    Returns:
        dumps the pickle file.
       """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)
def main():
    """
    Executes all the functions inside main to generate ML model pipeline
    
    Parameters:
        none
    
    Returns:
        none
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()
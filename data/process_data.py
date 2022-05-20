#importing libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Returns a merged python dataframe after reading two csv files and doing data mining.
    
    Parameters:
        messages_filepath (str): a string value for 'disaster_messages.csv' file path
        categories_filepath (str):a string value for 'disaster_categories.csv' file path
    
    Returns:
        df (python dataframe):a merged dataframe with messages and binary values from disaster_categories.csv
    """
    # load dataset
    categories = pd.read_csv(categories_filepath)
    messages = pd.read_csv(messages_filepath)
    messages = messages.drop_duplicates(subset=['id'])
    categories = categories.drop_duplicates(subset=['id'])
    #merge datasets
    df = messages.merge(categories, left_on='id', right_on='id', how='inner')
    df.drop_duplicates(subset=['id'],inplace=True)
    
    # create a dataframe of the 36 individual category columns
    categories = categories['categories'].str.split(";",expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[1]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = [(str(i)[:-2]) for i in row]
    print(category_colnames)

    # rename the columns of `categories`
    categories.columns = category_colnames
    #categories.head()

    for column in categories:
        # set each value to be the last character of the string

        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # drop the original categories column from `df`
    df.drop(columns=['categories'],axis=1,inplace=True)

    df.head()

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1).drop_duplicates()
    df.head()
    df.drop_duplicates(subset=['id'],inplace=True)
    df.drop(df[df['related'] == 2.0].index, inplace = True)
    df = df.fillna(0)
    return df
    pass


def clean_data(df):
    """
    Returns a cleaned python dataframe.
    
    Parameters:
        df(python dataframe): dataframe returned from load_data(messages_filepath,categories_filepath) function
    
    Returns:
        df (python dataframe):a cleaned dataframe after data mining and dropping duplicates.
    """
    df = df.drop_duplicates() #drop duplicates
    return df


def save_data(df, database_filename):
    """
    Returns a python dataframe after saving into a sqlite database.
    
    Parameters:
        df (python dataframe): dataframe returned from clean_data(df) function
        database_filename (str):a string value for database filename 
    
    Returns:
        df (python dataframe):a dataframe saved into DisasterResponse sqlite database 
       """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('DisasterResponse',engine,if_exists ='append', index=False)
    return df  


def main():
    """
    Executes all the functions inside main to generate ETL pipeline
    
    Parameters:
        none
    
    Returns:
        none
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
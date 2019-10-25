# import libraries

import argparse
import pandas as pd
import sqlite3
import sqlalchemy
from sqlalchemy import create_engine
import time
import numpy as np
import re

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string

# from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
# from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin

# from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.multioutput import MultiOutputClassifier
# from sklearn.externals import joblib

import cloudpickle
# import Joblib from joblib
from joblib import wrap_non_picklable_objects
import joblib
from sklearn.externals.joblib import parallel_backend

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


'''
Run the script like this:
python3 train_classifier.py 'message_cleaned.db' 'trained_rf_model.pkl'
'''

###################################
##### DATA LOADING FUNCTION ######
###################################

def load_sql_data(sql_file_path, table_name = 'message_cleaned'):
    '''
    This function loads the sqlite3 file.

    input:
    sql_file_path - sql file

    Output:
    X - messages (features)
    y - 36 message categories (labels)
    '''

    engine = create_engine('sqlite:///'+ sql_file_path)
    df = pd.read_sql_table(table_name, engine)
    X = df.loc[:, 'message'].values
    y = df.iloc[:, 4:].values

    labels = df.iloc[:, 4:].columns.values

    return X, y, labels

###################################
##### TOKNIZE TEXT DATA##### ######
###################################
#
def tokenize(text):
    '''
    This functions tokenize messages.

    Input:
    text - messages

    Output:
    toklist - a list of words with cleaned tokenized words (stopwords and punctuations removed)
    '''

    # Find urls in the message and remove them
    url_pattern = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    message_urls = re.findall(url_pattern, text)  # find urls
    for url in message_urls:
        text = text.replace(url, '')

    # Tokenize words
    tokens = word_tokenize(text)
    # Innitalize a lemmatizer
    lemmatizer = WordNetLemmatizer()
    # Get punctuation
    punct = [p for p in string.punctuation]

    toklist = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()

        words = stopwords.words('english')
        # Remove stop words and punctuations
        if clean_tok not in words and clean_tok not in punct:
            toklist.append(clean_tok)
    return ' '.join(toklist)

    # return toklist

# class TextTokenizer(BaseEstimator, TransformerMixin):
#     '''
#     This class toknize the words in messages, remove stopwords, normalize and remove white spaces.
#     '''
#
#     # def __init__(self):
#     #     pass
#
#
#
#     def transform(self, X, y = None):
#         def tokenize(text):
#             # Define url pattern
#             # url_re = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
#
#             # Detect and replace urls
#             # detected_urls = re.findall(url_re, text)
#             # for url in detected_urls:
#             #     text = text.replace(url, "urlplaceholder")
#
#             tokens = word_tokenize(text)
#             # Innitalize a lemmatizer
#             lemmatizer = WordNetLemmatizer()
#             # Get punctuation
#             punct = [p for p in string.punctuation]
#
#             # create a list of cleaned toknized words
#             toklist = []
#             for tok in tokens:
#                 clean_tok = lemmatizer.lemmatize(tok).lower().strip()
#
#                 words = stopwords.words('english')
#                 # Remove stop words and punctuations
#                 if clean_tok not in words and clean_tok not in punct:
#                     toklist.append(clean_tok)
#                 return ' '.join(toklist)
#             # Apply the toknized function to messages
#             return pd.Series(X).apply(tokenize).values
#
#     def fit(self, X, y=None):
#         return self

###################################
##### BUILDING MODEL FUNCTION #####
###################################

def build_model():
    """
    Build the pipeline model that is going to be used as the model
    """
    # popular_words = word_dict

    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer()),
    # ('clf', MultiOutputClassifier(RandomForestClassifier())),
    ('clf', MultiOutputClassifier(AdaBoostClassifier())),

    ])

    parameters = {'clf__estimator__n_estimators':[10,20]}

    cv = GridSearchCV(pipeline, param_grid = parameters, verbose=4) # test
    return cv


###################################
###### MODEL RUNNING FUNCTION #####
###################################

def train_model(model, X_train, y_train):
    '''
    This function takes features and labels data and train with a model.

    Input:
    X - messages (features)
    y - 36 message categories (labels)

    Output:
    model - model trained using the estimator and Parameters
    y_test - test outcome data
    y_pred - predicted outcome data produced by the model

    '''
    # train test split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # fit model

    start = time.time()
    model.fit(X_train, y_train)

    print(f'\nFinished training a model2. Took {(time.time() - start)/3600: .2f} hours')
    print("\nBest Parameters:", model.best_params_)

    return model


###################################
####### DISPLAYING FUNCTION #######
###################################

def evaluate_model(model, X_test, y_test, labels):
    '''
    This function displays the evaluation metrics - precision, recall, f1 score

    Input:
    model - model trained using the estimator and Parameters
    y_test - test outcome data
    y_pred - predicted outcome data produced by the model

    Output:
    prints the following - Confusion matrix, F1 score, precison, recall, and accuracy
    '''

    # predict category values
    y_preds = model.predict(X_test)
    # print metrics for each category
    for idx in range(0, len(labels)):
        classification_rpt = classification_report(y_test[:, idx], y_preds[:, idx])
        print(f'Category: {labels[idx]} - Classification Report\n', classification_rpt)

###################################
###### MODEL SAVING FUMCTION ######
###################################

def export_model(model, pkl_file_path):
    '''
    This function saves the model to a pickle file
    '''

    joblib.dump(model.best_estimator_, pkl_file_path)


#########################################
############# Main Function #############
#########################################

def main(sql_file_path, pkl_file_path):

    # print("Loading a sql file...")
    print("Loading data...")
    X, y, labels = load_sql_data(sql_file_path)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)
    # Build model pipeline
    print("Building a model...")
    model = build_model()

    # Fir model and make predictions
    print("\nTraining a model...")
    trained_model = train_model(model, X_train, y_train)

    # Saving the model as a pickle file
    export_model(trained_model, pkl_file_path)
    print("\nTrained model exported to a pickle file.")

    # Evaluate the mkodel and displays precision, recall, and f1 scores
    print("\nEvaluating model and isplaying results...")
    evaluate_model(trained_model, X_test, y_test, labels)

    print("\nDone!")


# def main(sql_file_path, pkl_file_path):
#
#     print("Loading SQL database...")
#     X, y, labels = load_sql_data(sql_file_path)
#
#     # Split the data into training and test datasets
#     X_train, X_test, y_train, y_test = train_test_split(X, y)
#
#     # Build model pipeline
#     print("Building a model...")
#     model = build_model()
#
#     # Fit model and make predictions
#     print("\nTraining a model...")
#     start = time.time()
#     model.fit(X_train, y_train)
#     y_preds = model.predict(X_test)
#
#     # print(f'\nFinished training a model. Took {(time.time() - start)/3600} hours')
#     print(f'\nFinished training a model2. Took {(time.time() - start)/3600: .2f} hours')
#     print("\nBest Parameters:", model.best_params_)
#
#     joblib.dump(model.best_estimator_, pkl_file_path)
#     print("\nTrained model exported to a pickle file.")
#
#     print("\nDisplaying results...")
#     display_results(y_test, y_preds, labels)
#     print("\nDone!")


#########################################
############ Set Up Objects #############
#########################################

## Parse in Arguments ##
parser = argparse.ArgumentParser()
parser.add_argument('sql_file_path', help = "path to the sql file")
parser.add_argument('pkl_file_path', help = "path to save the pickle file")
args = parser.parse_args()

if __name__ == "__main__":
    main(args.sql_file_path, args.pkl_file_path)

    # tokenize = TextTokenizer()
    # TextTokenizer.__module__ = 'train_classifier'
    # toknize.save('pkl_file_path')

README.md - Figure 8 Disaster Response Message Categorization


## **Project Title**

**Disaster Response Message Categorization (Udacity Data Science Course project)**


## **Description**

The goal of this project is to train the machine learning algorithm to categorize incoming disaster-related messages into 36 categories and to display the genres of messages message categories on the web application. The data was provided by FigureEight and included tweets and text sent during real disasters. 

**Pocesses**



1. **ETL Data Preparation**
*   This script merges the messages and categories data files and split the categories column which includes all categories into 36 separate columns.
*   Duplicate entries are removed.
*   The cleaned merged dataset is saved as a SQLite database.
*   To run process_data.py: _python3 ‘messages.csv’ ‘categoris.csv’ ‘messages_cat.db’ ‘message_table’_

2. **ML Pipeline and Model Optimization**
*   This scriot trains the machine learning model with multi-outputs by using gridsearch and stores the best model into a pickle file. 
*   The TF-IDF pipeline is only trained with training set. 
*   The model evaluation metrics are displayed for each category, including f1 score, precision and recall for the test set.  
*   To run train_classifier.py: _python3 train_classifier.py 'message_cat.db' 'trained_rf_model.pkl'_

3. **Web Application**
*   This script loads the SQLite database and pickle file and displays the two graphs (message genres and categorization). 
*   The user can enter the message into app and get the results for all 36 categories.
*   Run run.py as follows: _python3 run.py _


**Dataset Used:**

Two files, messages.csv and categories.csv, from Figure 8.

**Model Used:**

Random Forest algorithm with hyper parameters


## **Dependencies and Installation**

You need to install the following python, nlp, web, and visualization packages and libraries:

json, plotly, pandas, nltk, flask, sklearn, sqlalchemy, sys, numpy, re, pickle



*   scikit-learn: pip install scikit-learn
*   flask: pip install flask
*   plotly: pip install plotly
*   nltk: pip install nltk
*   nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
*   nltk.download (‘stopwords’) 
*   matplot: pip install matplotlib
*   seaborn: pip install seaborn
*   Joblib: pip install joblib


## **Acknowledgments**



*   Udacity course materials


<!-- Docs to Markdown version 1.0β17 -->

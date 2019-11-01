README.md - Figure 8 Disaster Response Message Classification
 
 
## **Project Title**
 
**Disaster Response Message Classification**
 
 
 
## **Description**
 
The goal of this project is to train the machine learning algorithm to categorize incoming disaster-related messages into 36 categories and to display the genres of messages message categories on the web application. The data was provided by FigureEight and included tweets and text sent during real disasters. 
 
**Pocesses**
 
1. **ETL Data Preparation**
*   This script merges the disaster messages and categories data files and split the categories column (which includes all categories) into 36 separate columns.
*   Duplicate entries are removed.
*   The cleaned merged dataset is saved as a SQLite database.
*   To run process_data.py: _<code>python3 ‘messages.csv’ ‘categoris.csv’ ‘messages_cleaned.db’ ‘message_table’</code>_
 
2. **ML Pipeline and Model Optimization**
*   This scriot trains the machine learning model with multi-outputs by using gridsearch and stores the best model into a pickle file. 
*   The TF-IDF pipeline is only trained with training set. 
*   The model evaluation metrics are displayed for each category, including f1 score, precision and recall for the test set.  
*   To run train_classifier.py: _<code>python3 train_classifier.py 'message_cat.db' 'trained_rf_model.pkl'</code>_
 
3. **Deployment**
*   This script loads the SQLite database and pickle file and displays the two graphs (message genres and categorization). 
*   The user can enter the message into app and get the results for all 36 categories.
*   To run run.py: _<code>python3 run.py </code>_
*   To see the web app: <code>http://0.0.0.0:3001/</code>
 
 
**Dataset Used:**
 
Two files, messages.csv and categories.csv, provided by Figure 8.
 
**Model Used:**
 
Random Forest algorithm was used to train the model, with TFIDF Vectorizer for text classification. 
 
 

## **Web App**
<p align="center"><code>Distribution of Message Genres and Categories</code></center></p>

![Alt text](/data/disaster_response_app_img1.png?raw=true "Distribution of Message Genres and Categories")

<p align="center"><code>Sample Message Classification</code></center></p>

![Alt text](/data/disaster_response_app_img2.png?raw=true "Message Classification ")

 
 
## **Dependencies and Installation**
 
You need to install the python, nlp, web, and visualization packages and libraries. They can be installed by running:
 
_<code>pip install -r requirements.txt</code>_ 


 
## **Acknowledgments**
 
 
 
This project was completed as part of the Udacity Data Scientist Nanodegree program. Course materials and starter code for web application were used to complete the project.
 
 
<!-- Docs to Markdown version 1.0β17 -->

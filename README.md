README.md - Figure 8 Disaster Response Message Classification


## **Project Title**

**Disaster Response Message Classification (Udacity Data Science Course project)**


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
*   Run run.py as follows: <code>_python3 run.py _</code>


**Dataset Used:**

Two files, messages.csv and categories.csv, from Figure 8.

**Model Used:**

Random Forest algorithm with hyper parameters


**App Images**

<p style="color: red; font-weight: bold">>>>>>  gd2md-html alert:  ERRORs: 0; WARNINGs: 0; ALERTS: 1.</p>
<ul style="color: red; font-weight: bold"><li>See top comment block for details on ERRORs and WARNINGs. <li>In the converted Markdown or HTML, search for inline alerts that start with >>>>>  gd2md-html alert:  for specific instances that need correction.</ul>

<p style="color: red; font-weight: bold">Links to alert messages:</p><a href="#gdcalert1">alert1</a>

<p style="color: red; font-weight: bold">>>>>> PLEASE check and correct alert issues and delete this message and the inline alerts.<hr></p>



<p id="gdcalert1" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/Disaster-Message0.png). Store image on your image server and adjust path/filename if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert2">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/Disaster-Message0.png "image_tooltip")


<p style="color: red; font-weight: bold">>>>>>  gd2md-html alert:  ERRORs: 0; WARNINGs: 0; ALERTS: 1.</p>
<ul style="color: red; font-weight: bold"><li>See top comment block for details on ERRORs and WARNINGs. <li>In the converted Markdown or HTML, search for inline alerts that start with >>>>>  gd2md-html alert:  for specific instances that need correction.</ul>

<p style="color: red; font-weight: bold">Links to alert messages:</p><a href="#gdcalert1">alert1</a>

<p style="color: red; font-weight: bold">>>>>> PLEASE check and correct alert issues and delete this message and the inline alerts.<hr></p>



<p id="gdcalert1" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/Disaster-Message0.png). Store image on your image server and adjust path/filename if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert2">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/Disaster-Message0.png "image_tooltip")
## **Dependencies and Installation**

You need to install the python, nlp, web, and visualization packages and libraries. they can be installed by running:

run pip install -r requirements.txt. 




## **Acknowledgments**



*   Udacity course materials


<!-- Docs to Markdown version 1.0β17 -->

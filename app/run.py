import json
import plotly
import pandas as pd
import numpy as np
import re
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    '''
    This functions removes urls and tokenize messages.

    Input:
    text - messages

    Output:
    toklist - a list of words with cleaned tokenized words (stopwords and punctuations removed)
    '''

    # Find urls in the message and remove them
    url_pattern = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    message_urls = re.findall(url_pattern, text)  # find urls
    for url in message_urls:
        text = text.replace(url, 'urlplaceholder')

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


# load data
engine = create_engine('sqlite:///../data/message_cleaned.db') # not tokenized
df = pd.read_sql_table('message_cleaned', engine)
print(df.head())

# load model
model = joblib.load("../models/trained_ada_model.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Percentages and names of 36 categories
    cat_perc = (df.iloc[:, 4:].isin([1]).sum(axis=0)/df.shape[0]*100).sort_values(ascending = False)
    cat_names = list(df.iloc[:, 4:].columns)

    # Create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                ),
            ],

            'layout': {
                'title': 'Types of Message Genres',
                'yaxis': {
                    'title': "Counts"
                },
                'xaxis': {
                    'title': "Genres"
                }
            }
        },

        {
            'data': [
                Bar(
                    x=cat_names,
                    y=cat_perc
                ),
            ],

            'layout': {
                'title': 'Distribution of 36 Message Categories',
                'yaxis': {
                    'title': "Percentages (%)"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        }

    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

    print('graph rendered.')
# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()

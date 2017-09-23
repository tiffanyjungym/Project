from flask import Flask
from flask import request
from flask import render_template

from bokeh.embed import components
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show, output_file

from nltk.corpus import reuters
from string import punctuation
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.util import ngrams
from nltk.stem.snowball import SnowballStemmer
import nltk

import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
import pdb
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, cross_validation
from sklearn.metrics import confusion_matrix


df5s21=pd.read_csv("/Users/tiffany/PycharmProjects/FlaskTiffany/Data/df5_clean_1.csv")

# From 365 products, subset 75 products for NLP training(dftrainnlp). 289 for  machine learning (df_ml, includes dftestnlp).

# In[3]:
def runModel(input):
    a=input
    output=1
    if output==1:
        message="This item is a likely Best Seller!"
    if output == 0:
        message="This item is an unlikely candidate"
    price= 90
    keyword= "strong, vacuum"

    return message, price, keyword

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template("userform.html")

@app.route('/', methods=['POST'])
def my_form_post():
    input = request.form['text']
    message,price,keywords = runModel(input)
    # message,price,keywords,figure = runModel(input)
    # fig_script, fig_div = components(figure)
    # return render_template('patient.html', message=message, fig_script=fig_script, fig_div=fig_div)
    return render_template('fortunecookie.html', message=message, price=price, keywords=keywords)

if __name__ == '__main__':
    app.run() 

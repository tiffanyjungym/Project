from flask import Flask
from flask import request
from flask import render_template


from bokeh.embed import components
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show, output_file

# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Written as part of https://www.scrapehero.com/how-to-scrape-amazon-product-reviews-using-python/
import nltk
import gzip
import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import HTML
import pdb
# import plotly.plotly as py
# import plotly.graph_objs as go
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, cross_validation
from sklearn.metrics import confusion_matrix
from lxml import html

import requests
import json, re
from dateutil import parser as dateparser
from time import sleep

from gensim import models
import gensim
from gensim.models.doc2vec import TaggedDocument
import pickle


# import json
# from pprint import pprint
# import re
# import pandas as pd

def ParseReviews(asin):
    # Added Retrying
    for i in range(5):
        try:
            # This script has only been tested with Amazon.com
            amazon_url = 'http://www.amazon.com/dp/' + asin
            # Add some recent user agent to prevent amazon from blocking the request
            # Find some chrome user agent strings  here https://udger.com/resources/ua-list/browser-detail?browser=Chrome
            headers = {
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.90 Safari/537.36'}
            page = requests.get(amazon_url, headers=headers)
            page_response = page.text

            parser = html.fromstring(page_response)
            XPATH_AGGREGATE = '//span[@id="acrCustomerReviewText"]'
            XPATH_REVIEW_SECTION_1 = '//div[contains(@id,"reviews-summary")]'
            XPATH_REVIEW_SECTION_2 = '//div[@data-hook="review"]'

            XPATH_AGGREGATE_RATING = '//table[@id="histogramTable"]//tr'
            XPATH_PRODUCT_NAME = '//h1//span[@id="productTitle"]//text()'
            XPATH_PRODUCT_PRICE = '//span[@id="priceblock_ourprice"]/text()'
            XPATH_CATEGORY = '//a[@class="a-link-normal a-color-tertiary"]/text()'

            raw_product_price = parser.xpath(XPATH_PRODUCT_PRICE)
            product_price = ''.join(raw_product_price).replace(',', '')
            raw_product_name = parser.xpath(XPATH_PRODUCT_NAME)
            product_name = ''.join(raw_product_name).strip()
            raw_categories = parser.xpath(XPATH_CATEGORY)
            categories = ''.join(raw_categories).replace('\n            \n            ', ',').strip()
            # 			categories=categories1.strip().split(',')
            total_ratings = parser.xpath(XPATH_AGGREGATE_RATING)
            reviews = parser.xpath(XPATH_REVIEW_SECTION_1)
            if not reviews:
                reviews = parser.xpath(XPATH_REVIEW_SECTION_2)
            ratings_dict = {}
            reviews_list = []

            if not reviews:
                raise ValueError('unable to find reviews in page')

            # grabing the rating  section in product page
            for ratings in total_ratings:
                extracted_rating = ratings.xpath('./td//a//text()')
                if extracted_rating:
                    rating_key = extracted_rating[0]
                    raw_raing_value = extracted_rating[1]
                    rating_value = raw_raing_value
                    if rating_key:
                        ratings_dict.update({rating_key: rating_value})
            # Parsing individual reviews
            for review in reviews:
                XPATH_RATING = './/i[@data-hook="review-star-rating"]//text()'
                XPATH_REVIEW_HEADER = './/a[@data-hook="review-title"]//text()'
                XPATH_REVIEW_POSTED_DATE = './/a[contains(@href,"/profile/")]/parent::span/following-sibling::span/text()'
                XPATH_REVIEW_TEXT_1 = './/div[@data-hook="review-collapsed"]//text()'
                XPATH_REVIEW_TEXT_2 = './/div//span[@data-action="columnbalancing-showfullreview"]/@data-columnbalancing-showfullreview'
                XPATH_REVIEW_COMMENTS = './/span[@data-hook="review-comment"]//text()'
                XPATH_AUTHOR = './/a[contains(@href,"/profile/")]/parent::span//text()'
                XPATH_REVIEW_TEXT_3 = './/div[contains(@id,"dpReviews")]/div/text()'
                raw_review_author = review.xpath(XPATH_AUTHOR)
                raw_review_rating = review.xpath(XPATH_RATING)
                raw_review_header = review.xpath(XPATH_REVIEW_HEADER)
                raw_review_posted_date = review.xpath(XPATH_REVIEW_POSTED_DATE)
                raw_review_text1 = review.xpath(XPATH_REVIEW_TEXT_1)
                raw_review_text2 = review.xpath(XPATH_REVIEW_TEXT_2)
                raw_review_text3 = review.xpath(XPATH_REVIEW_TEXT_3)

                author = ' '.join(' '.join(raw_review_author).split()).strip('By')

                # cleaning data
                review_rating = ''.join(raw_review_rating).replace('out of 5 stars', '')
                review_header = ' '.join(' '.join(raw_review_header).split())
                review_posted_date = dateparser.parse(''.join(raw_review_posted_date)).strftime('%d %b %Y')
                review_text = ' '.join(' '.join(raw_review_text1).split())

                # grabbing hidden comments if present
                if raw_review_text2:
                    json_loaded_review_data = json.loads(raw_review_text2[0])
                    json_loaded_review_data_text = json_loaded_review_data['rest']
                    cleaned_json_loaded_review_data_text = re.sub('<.*?>', '', json_loaded_review_data_text)
                    full_review_text = review_text + cleaned_json_loaded_review_data_text
                else:
                    full_review_text = review_text
                if not raw_review_text1:
                    full_review_text = ' '.join(' '.join(raw_review_text3).split())

                raw_review_comments = review.xpath(XPATH_REVIEW_COMMENTS)
                review_comments = ''.join(raw_review_comments)
                review_comments = re.sub('[A-Za-z]', '', review_comments).strip()
                review_dict = {
                    'review_comment_count': review_comments,
                    'review_text': full_review_text,
                    'review_posted_date': review_posted_date,
                    'review_header': review_header,
                    'review_rating': review_rating,
                    'review_author': author

                }
                reviews_list.append(review_dict)

            data = {
                'ratings': ratings_dict,
                'reviews': reviews_list,
                'url': amazon_url,
                'price': product_price,
                'name': product_name,
                'categories': categories
            }
            return data
        except ValueError:
            print "Retrying to get the correct response"

    return {"error": "failed to process the page", "asin": asin}


def ReadAsin(asinList):
    # Add your own ASINs here
    asinList = asinList
    scrapedData = []
    for asin in asinList:
        print "Downloading and processing page http://www.amazon.com/dp/" + asin
        scrapedData.append(ParseReviews(asin))
        sleep(5)
    return scrapedData


def GetDataInput(scrapedData, asinList):
    metaList = []
    reviewList = []
    starList = ['1 star', '2 star', '3 star', '4 star', '5 star']
    overall = 0

    tags = pd.Series()
    for i in range(0, len(scrapedData)):
        overall = 0
        asin = asinList[i]
        price = re.sub('[!@#$%]', '', scrapedData[i]['price'])
        for z in range(0, len(starList)):
            if starList[z] in scrapedData[i]['ratings']:
                overall = overall + (z + 1) * float(re.sub('[!@#$%]', '', scrapedData[i]['ratings'][starList[z]])) / 100
        description = scrapedData[i]['name']
        categories = pd.Series(scrapedData[i]['categories'])
        metaList.append((asin, description, price, overall, categories))

        for j in range(0, len(scrapedData[i]['reviews'])):
            reviewText = scrapedData[i]['reviews'][j]['review_text']
            reviewTime = scrapedData[i]['reviews'][j]['review_posted_date']
            reviewList.append((asin, reviewTime, reviewText))

    metaDfIn = pd.DataFrame(metaList, columns=['asin', 'description', 'price', 'overall', 'categories'])
    reviewDfIn = pd.DataFrame(reviewList, columns=['asin', 'reviewTime', 'reviewText'])
    dfIn = reviewDfIn.join(metaDfIn.set_index('asin'), on='asin')

    # Categories
    #     tags = pd.Series(dfIn['categories'])
    #     print tags
    #     tags2 = tags.rename(columns = lambda x : 'cat_' + str(x))
    #     dfIn=pd.concat([dfIn[:], tags2[:]], axis=1)
    #     print list(dfIn)

    # extract 5 reviews

    dfIn['reviewTime'] = pd.to_datetime(dfIn['reviewTime'])
    dfIns = dfIn.sort_values(['asin', 'reviewTime'], ascending=[True, True])
    dfIn5 = dfIns.groupby('asin').head(5).reset_index(drop=True)

    dfIn5['lenReviewText'] = dfIn5['reviewText'].str.len()
    dfIn5['lenDescription'] = dfIn5['description'].str.len()
    dfIn5['lenReviewText'] = dfIn5['lenReviewText'].fillna(0)
    dfIn5['lenDescription'] = dfIn5['lenDescription'].fillna(0)
    dfIn5['lenReviewTextAvg'] = dfIn5['lenReviewText'].groupby(dfIn5['asin']).transform("mean")
    dfIn5['lenDescriptionAvg'] = dfIn5['lenDescription'].groupby(dfIn5['asin']).transform("mean")

    # Unix time 86400 seconds/ day
    # list(dfIn5.columns.values)

    # maxtime=dfIn5['unixReviewTime'].max()
    dfIn5['numDaysPriorMax'] = 365
    dfIn5 = dfIn5.assign(
        daysToFiveRev=-dfIn5.sort_values('reviewTime', ascending=True).groupby(['asin']).reviewTime.diff(
            -4).dt.days.fillna(0))

    dfIn5['reviewText'] = dfIn5['reviewText'].apply(lambda x: x.encode('utf-8').strip())
    dfIn5['allReview'] = dfIn5.groupby(['asin'])['reviewText'].transform(lambda x: ', '.join(x))

    dfIn51 = dfIn5.groupby('asin').head(1).reset_index(drop=True)
    dfIn51 = dfIn51.fillna(0)

    return dfIn51

def RunDoc2Vec(dataInput):
    model= models.Doc2Vec.load('my_model.doc2vec')
    d2vtest=dataInput
    d2vlist=[]
    for i in range(0,len(d2vtest)):
        line=d2vtest.iloc[i]['reviewText']
        asin=d2vtest.iloc[i]['asin']
        tokens = gensim.utils.to_unicode(line).split()
        new_vector = model.infer_vector(tokens)
        topSim = model.docvecs.most_similar([new_vector])[0][0]
        secSim = model.docvecs.most_similar([new_vector])[1][0]
        lowSim = model.docvecs.most_similar([new_vector])[2][0]
        topSimSco = model.docvecs.most_similar([new_vector])[0][1]
        secSimSco = model.docvecs.most_similar([new_vector])[1][1]
        lowSimSco = model.docvecs.most_similar([new_vector])[2][1]
        d2vlist.append((asin,topSim,secSim,lowSim,topSimSco,secSimSco,lowSim))
        d2vresult=pd.DataFrame(d2vlist,columns=['asin','topSimilar','secSimilar','lowSimilar','scoSimHig','scoSimSec','scoSimLow'])
    #https://gist.github.com/balajikvijayan/9f7ab00f9bfd0bf56b14

    df_ml_d2v=pd.merge(d2vresult,d2vtest, on=['asin'])
    df_ml_d2v['scoDifHigLow']=df_ml_d2v['scoSimHig']-df_ml_d2v['scoSimLow']

    topDummy= pd.get_dummies(df_ml_d2v['topSimilar'], prefix='top')
    df_ml_d2v = pd.concat([df_ml_d2v, topDummy], axis=1, join_axes=[df_ml_d2v.index])
    return df_ml_d2v


def RunML(mlData):
    f = open('FortuneCookie.pickle', 'rb')
    trained_logistic_regression_model = pickle.load(f)
    f.close()

    training_features = ['topSimilar', 'scoDifHigLow', 'lenReviewTextAvg', 'lenDescriptionAvg',
                         'numDaysPriorMax', 'daysToFiveRev',
                         'overall', 'price',
                         'Canister Vacuums', 'Carpet Cleaners, Sweepers & Accessories', 'Handheld Vacuums',
                         'Robotic Vacuums', 'Stick Vacuums & Electric Brooms', 'Upright Vacuums',
                         'Bissell', 'Black &amp; Decker', 'Dirt Devil', 'Dyson', 'Electrolux',
                         'EnviroCare', 'Eureka', 'Euro-Pro', 'FilterStream', 'GV', 'Green Label', 'Hoover', 'Infinuvo',
                         'Irobot',
                         'Kenmore', 'Miele', 'Moneual', 'NEATO', 'Neato Robotics', 'Oreck',
                         'Oreck Merchandising LLC, us kitchen, OREBQ',
                         'Ovente', 'P3', 'Panasonic', 'Robot Add-Ons', 'Sebo Vacuums',
                         'Shark', 'Shop-Vac', 'Synergy', 'Techko Maid', 'The Bank Vacuum Company', 'Wrapables',
                         'iRobot']
    target = 'rankCat'

    for i in training_features:
        if i not in mlData:
            mlData[i] = 0
    mlData['prediction'] = trained_logistic_regression_model.predict(mlData[training_features])
    return mlData
    #     conf_matrix=confusion_matrix(test_y, test_x['prediction'])


#     sns.heatmap(conf_matrix, annot=True)
#     sns.plt.show()


app = Flask(__name__)


@app.route('/')
def my_form():
    return render_template("my-form.html")


@app.route('/', methods=['POST'])
def my_form_post():
    input = request.form['text'].replace(" ", "")
    asinList0 = list(input.split(','))
    asinList=[x.encode('UTF8') for x in asinList0]

    scrapedData = ReadAsin(asinList)
    dataInput = GetDataInput(scrapedData,asinList)
    mlData = RunDoc2Vec(dataInput)
    prediction = RunML(mlData)  # 0=best seller, 1=mid, 2=low

    predList = []
    for i in range(0, len(prediction)):
        asin = prediction.iloc[i]['asin']
        name = prediction.iloc[i]['description']
        dateFirst = prediction.iloc[i]['reviewTime']
        daysToFiveRev = prediction.iloc[i]['daysToFiveRev']
        reviewLength = prediction.iloc[i]['lenReviewTextAvg']
        price = prediction.iloc[i]['price']
        predRankCat = prediction.iloc[i]['prediction']
        predList.append((asin, name, dateFirst, daysToFiveRev, reviewLength, price, predRankCat))
        predResult = pd.DataFrame(predList,
                                  columns=['ASIN', 'Name', 'Date of first review', 'Time to five reviews (days)',
                                           'Average length of review', 'Price', 'Sales Rank'
                                           ])

    return render_template('fortunecookie.html', result=predResult.to_html(classes='table'))


if __name__ == '__main__':
    app.run() 

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


def run_model(input):
    df5s3=df5s21
    nlplen=df5s3.bestSell.value_counts()[1]-1
    ntrainnlp=0.5 # fraction of nlp samples

    ## NLP DATA SPLIT  by integer division (//)
    df5s3temp=df5s3.loc[(df5s3['bestSell'] == 1) | (df5s3['midSell'] == 1) | (df5s3['lowSell'] == 1)]

    # print df5s3.head(5)
    dftrainnlp0,dftestnlp=train_test_split(df5s3temp,test_size=ntrainnlp) #train, test for nlp
    df_ml = df5s3[~df5s3['asin'].isin(dftrainnlp0['asin'])]  ## Sample for ML

    trainhigh=dftrainnlp0[dftrainnlp0['bestSell']==1]['allReview'].str.cat(sep=', ')
    trainmid=dftrainnlp0[dftrainnlp0['midSell']==1]['allReview'].str.cat(sep=', ')
    trainlow=dftrainnlp0[dftrainnlp0['lowSell']==1]['allReview'].str.cat(sep=', ')

    dftrainnlp=pd.DataFrame({'reviewText':[trainhigh,trainmid,trainlow]})
    #
    # print len(dftrainnlp)
    # print len(df_ml)


    # Form training subsets and queries

    # In[ ]:

    # # WIthout cleaning (may be better https://groups.google.com/forum/#!topic/gensim/_7i926OSxl0)
    # from gensim import models
    # import gensim
    # from gensim.models.doc2vec import TaggedDocument

    # dfd2v=dftrainnlp
    # tags=['high','mid','low']
    # reviewLines={i:[] for i in range(len(dfd2v))}
    # sentences=[]

    # # print reviewLines
    # for i in range(0,len(dfd2v)):
    #     line=dfd2v.iloc[i]['reviewText']
    #     tokens = gensim.utils.to_unicode(line).split()
    #     words = tokens[1:]
    #     tags=[i]
    #     models.doc2vec.TaggedDocument(words=words,tags=[i])
    #     sentences.append(models.doc2vec.LabeledSentence(words=words, tags=[i]))

    # model = models.Doc2Vec(alpha=.025, min_alpha=.025, min_count=1)
    # model.build_vocab(sentences)

    # for epoch in range(10):
    #     model.train(sentences,total_examples=model.corpus_count,epochs=model.iter)
    #     model.alpha -= 0.002  # decrease the learning rate`
    #     model.min_alpha = model.alpha  # fix the learning rate, no decay

    # model.save('my_model.doc2vec')


    # In[6]:

    # WIthout cleaning (may be better https://groups.google.com/forum/#!topic/gensim/_7i926OSxl0)
    from gensim import models
    import gensim
    from gensim.models.doc2vec import TaggedDocument
    model= models.Doc2Vec.load('my_model.doc2vec')

    dfml=0
    d2vtest=df_ml
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
    ## Most similar, second similar, last similar

    # print d2vresult
    # print model_loaded.docvecs.most_similar(["B0084P56TI"])

    #https://gist.github.com/balajikvijayan/9f7ab00f9bfd0bf56b14

    df_ml_d2v=pd.merge(d2vresult,d2vtest, on=['asin'])
    df_ml_d2v['scoDifHigLow']=df_ml_d2v['scoSimHig']-df_ml_d2v['scoSimLow']



    topDummy= pd.get_dummies(df_ml_d2v['topSimilar'], prefix='top')
    df_ml_d2v = pd.concat([df_ml_d2v, topDummy], axis=1, join_axes=[df_ml_d2v.index])
    # print df_ml_d2v.head(5)

    conf_matrix=confusion_matrix(df_ml_d2v['bestSell'], df_ml_d2v['top_0'])
    sns.heatmap(conf_matrix, annot=True)
    # sns.plt.show()

    # sns.regplot('salesCat', 'topSimilar', data=dfml, x_estimator=None,
    #             x_bins=None, x_ci='ci', scatter=True, fit_reg=True, ci=95, n_boot=1000,
    #             units=None, order=1, logistic=False, lowess=False, robust=False, logx=False,
    #             x_partial=None, y_partial=None, truncate=False, dropna=True,
    #             x_jitter=None, y_jitter=None, label=None, color=None,
    #             marker='o', scatter_kws=None, line_kws=None, ax=None)
    # sns.plt.show()


    # In[57]:

    dataset = df_ml_d2v  ### df_ml_d2v: doc2vec cosine similarity input; df_ml_tf: tf_idf input
    train=0

    def train_logistic_regression(train_x, train_y):
        logistic_regression_model = LogisticRegression()
        logistic_regression_model.fit(train_x, train_y)
        print(logistic_regression_model.coef_)
        return logistic_regression_model


    def model_accuracy(trained_model, features, targets):
        accuracy_score = trained_model.score(features, targets)
        return accuracy_score

    # Load the data set for training and testing the logistic regression classifier

    # training_features = ['lenReviewTextAvg','lenDescriptionAvg','scoSimHig','scoDifHigLow','numDaysPriorMax','daysToFiveRev',
    #                      'overall','price',
    #                      'Canister Vacuums','Carpet Cleaners, Sweepers & Accessories','Handheld Vacuums',
    #                      'Robotic Vacuums','Stick Vacuums & Electric Brooms','Upright Vacuums',
    #                      'Bissell','Black &amp; Decker','Dirt Devil','Dyson','Electrolux',
    #                      'EnviroCare','Eureka','Euro-Pro','FilterStream','GV','Green Label','Hoover','Infinuvo','Irobot',
    #                      'Kenmore','Miele','Moneual','NEATO','Neato Robotics','Oreck','Oreck Merchandising LLC, us kitchen, OREBQ',
    #                      'Ovente','P3','Panasonic','Robot Add-Ons','Sebo Vacuums',
    #                      'Shark','Shop-Vac','Synergy','Techko Maid','The Bank Vacuum Company','Wrapables','iRobot'] #,'lenDescription','numDaysPrior','overall',

    # training_features = ['topSimilar','lenReviewTextAvg','lenDescriptionAvg','scoDifHigLow','numDaysPriorMax','daysToFiveRev']
    # training_features = ['topSimilar','lenReviewTextAvg','lenDescriptionAvg','scoDifHigLow','numDaysPriorMax','daysToFiveRev']

    training_features = ['topSimilar','scoDifHigLow','lenReviewTextAvg','lenDescriptionAvg',
                         'numDaysPriorMax','daysToFiveRev',
                         'overall','price',
                         'Canister Vacuums','Carpet Cleaners, Sweepers & Accessories','Handheld Vacuums',
                         'Robotic Vacuums','Stick Vacuums & Electric Brooms','Upright Vacuums',
                         'Bissell','Black &amp; Decker','Dirt Devil','Dyson','Electrolux',
                         'EnviroCare','Eureka','Euro-Pro','FilterStream','GV','Green Label','Hoover','Infinuvo','Irobot',
                         'Kenmore','Miele','Moneual','NEATO','Neato Robotics','Oreck','Oreck Merchandising LLC, us kitchen, OREBQ',
                         'Ovente','P3','Panasonic','Robot Add-Ons','Sebo Vacuums',
                         'Shark','Shop-Vac','Synergy','Techko Maid','The Bank Vacuum Company','Wrapables','iRobot']

    target = 'bestSell'

    # Train , Test data split
    train_ximbal, test_x, train_yimbal, test_y = train_test_split(dataset[training_features], dataset[target], train_size=0.7)
    df_ml_train=pd.concat([train_ximbal,train_yimbal], axis=1)


    ## UPSAMPLING
    from sklearn.utils import resample
    nsamples=int(len(df_ml_train)*0.85)
    #int(len(dfml)*0.9)
    dfimbal=df_ml_train
    df_majority = dfimbal[dfimbal.bestSell==0]
    df_minority = dfimbal[dfimbal.bestSell==1]


    # Upsample minority class
    df_minority_upsampled = resample(df_minority,
                                     replace=True,     # sample with replacement
                                     n_samples=nsamples,    # to match majority class
                                     random_state=123) # reproducible results


    # Combine majority class with upsampled minority class
    train_upsampled = pd.concat([df_majority, df_minority_upsampled])
    train_y=train_upsampled['bestSell']
    train_x=train_upsampled[training_features]

    # print dfimbal.bestSell.value_counts()

    # Training Logistic regression model
    trained_logistic_regression_model = train_logistic_regression(train_x, train_y)
    train_accuracy = model_accuracy(trained_logistic_regression_model, train_x, train_y)


    # Testing the logistic regression model
    test_accuracy = model_accuracy(trained_logistic_regression_model, test_x, test_y)
    #     print "Train Accuracy :: ", train_accuracy
    #     print "Test Accuracy :: ", test_accuracy
    predicted = cross_validation.cross_val_predict(LogisticRegression(), train_x, train_y, cv=10)
    # print metrics.accuracy_score(train_y, predicted)
    # print metrics.classification_report(train_y, predicted)
    # print metrics.roc_auc_score(train_y, predicted)

    ## ROC Curve
    preds = trained_logistic_regression_model.predict_proba(test_x)[:,1]
    fpr, tpr, _ = metrics.roc_curve(test_y, preds)
    df_preds = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
    #     print fpr
    #     print tpr
    lw=2
    # plt.plot(fpr, tpr, color='darkorange',
    #          lw=lw) #, label='ROC curve (area = %0.2f)' % roc_auc[2]
    # plt.show()


    test_x['prediction']=trained_logistic_regression_model.predict(test_x)
    conf_matrix=confusion_matrix(test_y, test_x['prediction'])
    sns.heatmap(conf_matrix, annot=True)
    #sns.plt.show()


    # In[55]:


    # output=test_x.head(0)
    output=test_x['prediction'].iloc[0]

    if output==1:
        message1="You've got a best seller!"
        message2="This item is 50% more likely to be a best seller within 1 year."
    if output == 0:
        message1="Not a chance."
        message2="This item is only 2% likely to be a best seller."

    message2_1="You've got a best seller!"
    message2_2="This item is 50% more likely to be a best seller within 1 year."

    message3_1="Not a chance."
    message3_2="This item is only 5% likely to be a best seller."

    price= "10% quantile in the same category"
    keyword= "strong, vacuum"

    return message1, message2, message2_1, message2_2, message3_1, message3_2, price, keyword


app = Flask(__name__)


@app.route('/')
def my_form():
    return render_template("my-form.html")


@app.route('/', methods=['POST'])
def my_form_post():
    input = request.form['text']
    message1,message2,message2_1, message2_2,message3_1, message3_2, price,keyword = run_model(input)
    # fig_script, fig_div = components(figure)
    return render_template('fortunecookie.html', message1=message1, message2=message2,
                           message2_1=message2_1, message2_2=message2_2,
                           message3_1=message3_1,  message3_2=message3_2, price=price, keyword=keyword,
                           )


if __name__ == '__main__':
    app.run() 

import warnings
warnings.filterwarnings('ignore')
import os
import pickle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams["figure.figsize"] = (20, 10)
import re
import string
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from googletrans import Translator



from flask import Flask, redirect, url_for, request
app = Flask(__name__)



@app.route('/success/<text>')
def success(text):
    data = {}
    data[0] = text
    pd.set_option('max_colwidth', 200)
    df1 = pd.DataFrame.from_dict(data, orient='index')
    df1.columns = ['Lyrics']

    def round1(text):
        # lower the Text
        text = text.lower()
        # Remove Numbers
        text = re.sub(r"\d+", "", text)
        # Remove Symbols and special characters
        # Below return true if not alphanumereic
        text = re.sub(r'[^\w]', ' ', text)
        # Remove more than a single whitespace
        text = ' '.join(text.split())
        # Remove Leading and Trailing Whitespaces
        text = text.strip()
        return text

    rnd1 = lambda x: round1(x)
    df2 = df1.copy()
    df2['Lyrics'] = df2['Lyrics'].apply(rnd1)

    stop = list(string.punctuation)

    def cleaning(text):
        clean_doc = []
        for x in text:
            clean_sent = []
            for i in word_tokenize(x):
                # for i in x.lower():
                if i not in stop:
                    clean_sent.append(i)
            clean_doc.append(clean_sent)
        return clean_doc

    df3 = df2.copy()
    df3['Lyrics'] = cleaning(df3['Lyrics'])

    s = ' '
    for i in range(len(df3)):
        df3['Lyrics'].loc[i] = s.join(df3['Lyrics'].loc[i])

    wordnet = WordNetLemmatizer()

    def Lemmatizing(text):
        pre_doc = []
        for word in text:
            pre_doc.append(wordnet.lemmatize(word))
        return pre_doc

    df4 = df3.copy()
    df4['Lyrics'] = Lemmatizing(df4['Lyrics'])

    cv = CountVectorizer(stop_words='english')
    df5 = cv.fit_transform(df4['Lyrics'])
    df6 = pd.DataFrame(df5.toarray(), columns=cv.get_feature_names())
    df6.index = df4.index

    df7 = df6.transpose()

    top_dict = {}
    for c in df7.columns:
        top = df7[c].sort_values(ascending=False).head(30)
        top_dict[c] = list(zip(top.index, top.values))

    for album, top_words in top_dict.items():
        print(album)
        print(', '.join([word for word, count in top_words]))
        print('------------')

    ts = Translator()
    res = ts.translate(df4['Lyrics'].loc[0], dest='hi')
    hitext = res.text
    return '%s' % hitext

@app.route('/index', methods = ['POST', 'GET'])
def login():
   if request.method == 'POST':
      data = request.form['nm']
      return redirect(url_for('success', text = data))


if __name__ == '__main__':
   app.run(debug = True)
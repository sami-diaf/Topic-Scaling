"""
@author: Sami Diaf
"""

""" Dynamic Topic Model (per decade) for SOTU Addresses"""

import pandas as pd
import numpy as np

import gensim
import os

from gensim.test.utils import common_corpus, common_dictionary
from gensim.models.wrappers import DtmModel
from gensim.utils import simple_preprocess, lemmatize
from gensim import corpora, utils
from gensim.models.wrappers.dtmmodel import DtmModel

import re
import io
import logging
import math
import time
import string

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = stopwords.words('english')

# getting the dataset
url = "https://raw.githubusercontent.com/BrianWeinstein/state-of-the-union/master/transcripts.csv"
data=pd.read_csv(url, sep = ",")

data = data.sort_values(by=['date'])

# creating the year variable
data['year'] =data['date'].apply(lambda x: time.strptime(x, '%Y-%m-%d').tm_year) 

data['year'].value_counts()

# cleaning/pre-processing text data
data['txt'] = data['transcript'].str.replace('[^\w\s]',' ')
data['txt']= data['txt'].str.rstrip(string.digits)

# selecting tokens
data['tok']=data['txt'].apply(lambda x: x.lower().split() if x not in stop_words and len(x) >2 else x)   

documents = data['tok'].tolist()

for i in range(len(data.txt)):
    data['txt'][i] = data['txt'][i].lower()
    
data.txt = data.txt.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
data.txt = data.txt.apply(lambda x: ' '.join([word for word in x.split() if len(word) > 1]))

docs = [x.split() for x in data.txt]

# creating documents for DTM corpus    
docs_lem=[]
for i in docs:
    a=[j for j in i]
    docs_lem.append(a)
    
    
class DTMcorpus(corpora.textcorpus.TextCorpus):

    def get_texts(self):
        return self.input

    def __len__(self):
        return len(self.input)

corpus = DTMcorpus(docs_lem)

# creating decade variable that serves as a time frame for DTM
data['decade'] = (data['year']-1850)/10
data['decade'] = data['decade'] .apply(np.floor)

time_slices = list(data['decade'].value_counts())[::-1]

# dtm path 
dtm_path = "/home/sami/dtm/dtm/main"

# estimating DTM with 2 topics per decade
model = DtmModel(dtm_path, corpus, time_slices, num_topics=2,
                 id2word=corpus.dictionary)

# displaying top 10 words in topic number 1 during the second decade
model.show_topic(topicid=1, time=1, topn=10)

doc_number = 0
num_topics = 2

# topic distribution during first time frame (decade)
for i in range(0,num_topics):
    print("Distribution of topic %d %f" % (i, model.gamma_[doc_number, i]))

results = pd.DataFrame()

for i in range(0,num_topics):
    results[i+1]=model.gamma_[data.index,i]
    
    
    
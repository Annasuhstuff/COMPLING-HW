#!/usr/bin/env python
# coding: utf-8

# In[342]:


from IPython.display import Image
from IPython.core.display import HTML 
Image(url= "https://www.meme-arsenal.com/memes/a33d72b526c94cc9b3f4b22044324dab.jpg")


# 1 Задание

# In[217]:


import pandas as pd


# In[216]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, f1_score
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity

from IPython.display import Image
from IPython.core.display import HTML 

from razdel import tokenize
from nltk.corpus import stopwords
import numpy as np
import string


# In[218]:


from razdel import tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# In[219]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score


# In[220]:


import math 


# In[221]:


import nltk


# In[222]:


nltk.download("stopwords")
russian_stopwords = stopwords.words("russian") 


# In[223]:


data = pd.read_csv('labeled.csv', index_col=False)
x_train, x_test, y_train, y_test = train_test_split(data['comment'], data['toxic'], test_size=0.25, shuffle=True)


# In[224]:


default_vectorizer = CountVectorizer()
x_train_tokens = default_vectorizer.fit_transform(x_train.values.tolist())
x_test_tokens = default_vectorizer.transform(x_test.values.tolist()) 


# In[225]:


def razdel_tokenizer(text):
    return [item.text for item in list(tokenize(text))]    
custom_vectorizer = CountVectorizer(tokenizer=razdel_tokenizer)
X_train_tokens = custom_vectorizer.fit_transform(x_train.values.tolist())
X_test_tokens = custom_vectorizer.transform(x_test.values.tolist()) 


# In[226]:


logRegDefault = LogisticRegression(max_iter=1400)
logRegRazdel = LogisticRegression(max_iter=1400)
logRegDefault.fit(x_train_tokens, y_train)
logRegRazdel.fit(X_train_tokens, y_train)


# Итак, попробуем измерить accuracy с помощью логистической регрессии:

# In[227]:


print("Default accuracy - ", logRegDefault.score(x_test_tokens, y_test))
print("Razdel accuracy - ", logRegRazdel.score(X_test_tokens, y_test))


# In[228]:


from sklearn.tree import DecisionTreeClassifier as DTC


# Разница совсем небольшая, разделовский токенайзер работает чуть лучше.

# Также можно воспользоваться деревом решений:

# In[229]:


# метрика для разделовского токенайзера
DTC = DecisionTreeClassifier()
DTC.fit(X_train_tokens, y_train)

DTC_preds = DTC.predict(X_test_tokens)
print(classification_report(y_test, DTC_preds))


# In[230]:


# метрика для дефолтного токенайзера
DTC.fit(x_train_tokens, y_train)


DTC_preds = DTC.predict(x_test_tokens)
print(classification_report(y_test, DTC_preds))


# Выводы по второй метрике: дефолтный токенайзер и разделовский токенайзер почти не отличаются, мы видим только небольшие различия.

# Итог: в принципе, если несколько раз от начала до конца делать RUN, то результаты метрики будут разными. Как я поняла, это из-за того, что в датасете может быть неравно сбалансированное количество токсичных и нетоксичных комментариев. 

# 2 задание

# In[346]:


Image(url= "http://www.quickmeme.com/img/6f/6f234c5b9c9feabe6da8f8a51fe49bc82e3f116287a526e7b6d5d4d34cc41dd1.jpg")


# In[233]:


Image(url="https://i.ibb.co/r5Nc2HC/abs-bow.jpg")


# In[234]:


f = [[ 1,1,1,0,0,0],[1,1,1,0,0,0], [3,0,1,1,0,0],[1,0,0,1,1,0], [0,0,0,0,0,1]]


# In[235]:


tablichka = pd.DataFrame(f, columns = ['я', 'ты', 'и', 'только', 'не', 'он'], index = ['я и ты','ты и я','я, я и только я', 'только не я', 'он'])


# ищем tf

# In[236]:



def count_tf(term, sent):
    tf = []
    tf = sent.count(term) / len(sent.split())
    return tf


# ищем df

# In[237]:


def count_idf(term, text): 
        df = 0
        idf = []
        for d in text:
            if term in d:
                df += 1
                idf = math.log(len(text) / df) 
        return idf


# tfidf

# In[238]:


def count_tfidf(term, sent, text): 
    tfidf = []
    tfidf = count_tf(term, sent) * count_idf(term, text)
    return tfidf


# In[239]:


tablichka_tfidf = pd.DataFrame()

for a in voc:
    x = []
    for b in text:
        x.append(count_tfidf(a, b, text))
    tablichka_tfidf[a] = x
    
tablichka_tfidf.index = text


# In[240]:


print (tablichka_tfidf)


# 3 задание

# In[319]:


x_train, x_test, y_train, y_test = train_test_split(data['comment'], data['toxic'], test_size=0.25, shuffle=True)

y = train.toxic.values
y_test = test.toxic.values


# In[334]:


vectorizer = CountVectorizer(max_df=0.1, min_df=5, stop_words=russian_stopwords , max_features=4321, ngram_range=(1, 2))

X = vectorizer.fit_transform(train.comment)
X_test = vectorizer.transform(test.comment)

clf = LogisticRegression(dual=False, tol=0.0001, C=0.1,solver='lbfgs', class_weight='balanced')

clf.fit(X, y)

predictions_1 = clf.predict(X_test)
proba_1 = clf.predict_proba(X_test)

f1_score(y_test, predictions_1)


# In[335]:


vectorizer = TfidfVectorizer(max_features=7000,min_df=5, max_df=0.6, ngram_range=(1, 2),tokenizer=razdel_tokenizer)

X = vectorizer.fit_transform(train.comment)
X_test = vectorizer.transform(test.comment) 

clf = MultinomialNB(alpha=0.1, fit_prior=False, class_prior=None)

clf.fit(X, y)

predictions_2 = clf.predict(X_test)
proba_2 = clf.predict_proba(X_test)

f1_score(y_test, predictions_2)


# In[338]:


def toxic(probas):
    probas = [p[1] for p in probas]
    test['probas'] = probas
    res_df = test.sort_values(by = 'probas', ascending = False)[:10]
    res_df = res_df.reset_index(drop=True)
    for i in range(10):
        print('toxic:', res_df.loc[i].toxic)
        print('comment: ', res_df.loc[i].comment)
    return res_df


# In[340]:


print('1-й классификатор')
clf1 = toxic(proba_1)
print('2-й классификатор')
clf2 = toxic(proba_2)


# In[347]:


Image(url= "https://i.ytimg.com/vi/7twy1gjLR6A/hqdefault.jpg")


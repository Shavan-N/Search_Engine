# Importing necessary libraries
import requests
from bs4 import BeautifulSoup
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from googletrans import Translator



# Extracting the documents from https://bola.kompas.com/
r = requests.get('https://bola.kompas.com/')
soup = BeautifulSoup(r.content, 'html.parser')
link = []
for i in soup.find('div', {'class': 'most__wrap'}).find_all('a'):
    i['href'] = i['href'] + '?page=all'
    link.append(i['href'])

documents = []
for i in link:
    r = requests.get(i)
    soup = BeautifulSoup(r.content, 'html.parser')
    sen = []
    for i in soup.find('div', {'class': 'read__content'}).find_all('p'):
        sen.append(i.text)
    documents.append(' '.join(sen))



# Cleaning the documents
documents_clean = []
for d in documents:
    document_test = re.sub(r'[^\x00-\x7F]+', ' ', d)
    document_test = re.sub(r'@\w+', '', document_test)
    document_test = document_test.lower()
    document_test = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', document_test)
    document_test = re.sub(r'[0-9]', '', document_test)
    document_test = re.sub(r'\s{2,}', ' ', document_test)
    documents_clean.append(document_test)



# Creating Term-Document Matrix with TF-IDF weighting
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents_clean)
X = X.T.toarray()
df = pd.DataFrame(X, index=vectorizer.get_feature_names())



# Calculating the similarity using cosine similarity
# Cosine = q * d / (|q| * |d|)
def get_similar_articles(q, df):
    print("query: ", q)
    print('The following is the article with the highest cosine similarity value:')
    q = [q]
    q_vec = vectorizer.transform(q).toarray().reshape(df.shape[0],)
    sim = {}
    for i in range(10):
        sim[i] = np.dot(df.loc[:, i].values, q_vec) / np.linalg.norm(df.loc[:, i]) * np.linalg.norm(q_vec)
    
    sim_sorted = sorted(sim.items(), key=lambda x: x[1], reverse=True)
    for k,v in sim_sorted:
        if v != 0.0:
            print('Similarity value: ', v)
            translator = Translator(service_urls=['translate.googleapis.com'])
            result = translator.translate(documents_clean[k], dest='en', src='id')
            print(result.text)
            print()



# Running queries
q1 = input('Enter keyword to search: ')
get_similar_articles(q1, df)
# print('\n\n\n\n', documents_clean)

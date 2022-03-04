# Importing necessary libraries
import requests
from bs4 import BeautifulSoup
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from googletrans import Translator
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import sent_tokenize , word_tokenize
import glob
import os
import sys
Stopwords = set(stopwords.words('english'))
print('Imported all the libraries')



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
print('Extracted all the articles')



# Cleaning the documents
documents_clean = []
for d in documents:
    document_test = re.sub(r'[^\x00-\x7F]+', ' ', d)
    document_test = re.sub(r'@\w+', '', document_test)
    document_test = document_test.lower()
    document_test = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', document_test)
    document_test = re.sub(r'[0-9]', '', document_test)
    document_test = re.sub(r'\s{2,}', ' ', document_test)
    translator = Translator(service_urls=['translate.googleapis.com'])
    result = translator.translate(document_test, dest='en', src='id')
    documents_clean.append(result.text)
print('Cleaning of articles completed')



# Saving the articles in 'data' folder
k = 1
for d in documents_clean:
    file_name = 'data/document_' + str(k) + '.txt'
    # print(file_name)
    k += 1
    file = open(file_name, 'w')
    file.write(d)
    file.close()
print('Saved the articles')



# Implementing Helper functoins
def finding_all_unique_words_and_freq(words):
    words_unique = []
    word_freq = {}
    for word in words:
        if word not in words_unique:
            words_unique.append(word)
    for word in words_unique:
        word_freq[word] = words.count(word)
    return word_freq

def finding_freq_of_word_in_doc(word,words):
    freq = words.count(word)
        
def remove_special_characters(text):
    regex = re.compile('[^a-zA-Z0-9\s]')
    text_returned = re.sub(regex,'',text)
    return text_returned
print('Implemented helper functions')



# Defining the linked list
class Node:
    def __init__(self ,docId, freq = None):
        self.freq = freq
        self.doc = docId
        self.nextval = None
    
class SlinkedList:
    def __init__(self ,head = None):
        self.head = head
print('Implemented Linked List Classes')



# Finding the set of unique words from all documents of the data set
all_words = []
dict_global = {}
file_folder = 'data/*'
idx = 1
files_with_index = {}
for file in glob.glob(file_folder):
    print(file)
    fname = file
    file = open(file , "r")
    text = file.read()
    text = remove_special_characters(text)
    text = re.sub(re.compile('\d'),'',text)
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    words = [word for word in words if len(words)>1]
    words = [word.lower() for word in words]
    words = [word for word in words if word not in Stopwords]
    dict_global.update(finding_all_unique_words_and_freq(words))
    files_with_index[idx] = os.path.basename(fname)
    idx = idx + 1
    
unique_words_all = set(dict_global.keys())
print('Created set of unique words from all documents')



# Making a linkedlist for each word and storing all the nodes
linked_list_data = {}
for word in unique_words_all:
    linked_list_data[word] = SlinkedList()
    linked_list_data[word].head = Node(1,Node)
word_freq_in_doc = {}
idx = 1
for file in glob.glob(file_folder):
    file = open(file, "r")
    text = file.read()
    text = remove_special_characters(text)
    text = re.sub(re.compile('\d'),'',text)
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    words = [word for word in words if len(words)>1]
    words = [word.lower() for word in words]
    words = [word for word in words if word not in Stopwords]
    word_freq_in_doc = finding_all_unique_words_and_freq(words)
    for word in word_freq_in_doc.keys():
        linked_list = linked_list_data[word].head
        while linked_list.nextval is not None:
            linked_list = linked_list.nextval
        linked_list.nextval = Node(idx ,word_freq_in_doc[word])
    idx = idx + 1
print('Created a Linked List')



# Boolean Query processing and output generation
query = input('Enter your query:')
query = word_tokenize(query)
connecting_words = []
cnt = 1
different_words = []
for word in query:
    if word.lower() != "and" and word.lower() != "or" and word.lower() != "not":
        different_words.append(word.lower())
    else:
        connecting_words.append(word.lower())
print(connecting_words)
total_files = len(files_with_index)
zeroes_and_ones = []
zeroes_and_ones_of_all_words = []
for word in (different_words):
    if word.lower() in unique_words_all:
        zeroes_and_ones = [0] * total_files
        linkedlist = linked_list_data[word].head
        print(word)
        while linkedlist.nextval is not None:
            zeroes_and_ones[linkedlist.nextval.doc - 1] = 1
            linkedlist = linkedlist.nextval
        zeroes_and_ones_of_all_words.append(zeroes_and_ones)
    else:
        print(word," not found")
        sys.exit()
print(zeroes_and_ones_of_all_words)
for word in connecting_words:
    word_list1 = zeroes_and_ones_of_all_words[0]
    word_list2 = zeroes_and_ones_of_all_words[1]
    if word == "and":
        bitwise_op = [w1 & w2 for (w1,w2) in zip(word_list1,word_list2)]
        zeroes_and_ones_of_all_words.remove(word_list1)
        zeroes_and_ones_of_all_words.remove(word_list2)
        zeroes_and_ones_of_all_words.insert(0, bitwise_op);
    elif word == "or":
        bitwise_op = [w1 | w2 for (w1,w2) in zip(word_list1,word_list2)]
        zeroes_and_ones_of_all_words.remove(word_list1)
        zeroes_and_ones_of_all_words.remove(word_list2)
        zeroes_and_ones_of_all_words.insert(0, bitwise_op);
    elif word == "not":
        bitwise_op = [not w1 for w1 in word_list2]
        bitwise_op = [int(b == True) for b in bitwise_op]
        zeroes_and_ones_of_all_words.remove(word_list2)
        zeroes_and_ones_of_all_words.remove(word_list1)
        bitwise_op = [w1 & w2 for (w1,w2) in zip(word_list1,bitwise_op)]
zeroes_and_ones_of_all_words.insert(0, bitwise_op);
        
files = []    
print(zeroes_and_ones_of_all_words)
lis = zeroes_and_ones_of_all_words[0]
cnt = 1
for index in lis:
    if index == 1:
        files.append(files_with_index[cnt])
    cnt = cnt+1
    
print(files)
print('Found all the documents which satisfy the query')
print('\n')



# Rank Retrieval Model
docs = []
for file in files:
    file_name = 'data/' + file
    file1 = open(file_name, 'r')
    docs.append(file1.read())



# Creating Term-Document Matrix with TF-IDF weighting
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs)
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
    for i in range(len(docs)):
        sim[i] = np.dot(df.loc[:, i].values, q_vec) / np.linalg.norm(df.loc[:, i]) * np.linalg.norm(q_vec)
    
    sim_sorted = sorted(sim.items(), key=lambda x: x[1], reverse=True)
    sim_val = 0
    final_ans = ''
    docs_name = ''
    for k,v in sim_sorted:
        # if v != 0.0:
            # print('Similarity value: ', v)
            # translator = Translator(service_urls=['translate.googleapis.com'])
            # result = translator.translate(documents_clean[k], dest='en', src='id')
            # print(result.text)
            # print(docs[k])
            # print()
        if v > sim_val:
            final_ans = docs[k]
            docs_name = k
    print('The highest similarity value is ', v)
    print(final_ans)
    print(docs_name)



# Running queries
q1 = different_words[0]
get_similar_articles(q1, df)
# print('\n\n\n\n', documents_clean)

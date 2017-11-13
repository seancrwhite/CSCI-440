#authors: Tia Smith, Sean White
#date modified: 11/10/17
#description: Find answers to questions about films by querying IMDb database and performing analysis and visualization using machine learning techniques

import csv
import sys
import re
import sklearn
import codecs
import numpy as np
import MySQLdb
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA

# remove special characters from text
def remove_text(text):
    return re.sub('-','',text)

# query database and load data into dictionary
def load_data(data):
    data = {} #initialize dictionary to hold data
    #add data to dictionary (SQL queries)
    return data

#generate features and labels
def make_dataset(data):
    labels = []; #initialize a labels list
    features = []
    count = countVectorizer()
    data = []
    #optional cleaning of words - we may want to omit for simplicity / efficiency
    stop = stopwords.words('english')
    porter = PorterStemmer()
    #do stopping and stemming
    bag = count.fit_transofrm(data) #generates bag of words
    features = bag
    return features, labels

#analyze and return auc
def cross_validation(name, features, labels, num_folds):
    if name.lower() == 'knn':
        data = cross_val_score(KNeighborsClassifier(), features, labels, cv=num_folds, scoring='roc_auc')
    elif name.lower() == 'gnb':
        data = cross_val_score(GaussianNB(), features.toarray(), labels, cv=num_folds, scoring='roc_auc')
    elif name.lower() == 'tree':
        data = cross_val_score(DecisionTreeClassifier(), features.toarray(), labels, cv=num_folds, scoring=='roc_auc')
    elif name.lower() == 'svc':
        data = cross_val_score(LinearSVC(), features.toarray(), labels, cv=num_folds, scoring='roc_auc')
    elif name.lower() == 'mlp':
        data = cross_val_score(MLPClassifier(), features.toarray(), labels, cv=num_folds, scoring='roc_auc')
    auc = np.mean(data)
    return auc

if __name__ == "__main__":
    question = sys.argv[1] #e.g. 1
    name = sys.argv[2] #e.g. knn

    db = MySQLdb.connect(port = 3306,
                         user = "root",
                         passwd = "password",
                         db = "IMDB")

    num_folds = int(sys.argv[3]) #e.g. 5
    data = load_data(data)
    features, labels = make_dataset(data)
    auroc = cross_validation(name, features, labels, num_folds)

    print("question: ", question)
    print("classifier: ", name)
    print("number of folds: ", num_folds)
    print("AUROC: ", auroc)

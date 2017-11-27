#authors: Tia Smith, Sean White
#description: Find answers to questions about films by querying IMDb database and performing analysis and visualization using machine learning techniques

import csv
import sys
import re
import sklearn
import codecs
import numpy as np
import operator
import heapq
import matplotlib.pyplot as plt
import mysql.connector as dbc
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import normalize
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA

# remove special characters from text
def remove_text(text):
    return re.sub('-','',text)

# query database and load data into dictionary
def load_data(question, cursor):
    query = "" # default empty query

    if question == 1:
        query = ("SELECT Gross_Revanue, Budget, Duration, Aspect_Ratio, Release_Year, Votes, IMDB_Score "
                 "FROM IMDB.EARNINGS NATURAL JOIN IMDB.SHOW NATURAL JOIN IMDB.SCORE "
                 "WHERE Gross_Revanue > 999999 AND Duration IS NOT NULL")
    elif question == 2:
        query = ("SELECT Show_Id, Title, Gross_Revanue, Name, Role "
                 "FROM IMDB.EARNINGS NATURAL JOIN IMDB.SHOW NATURAL JOIN "
                 "IMDB.WORKED_ON NATURAL JOIN IMDB.PERSON")
    elif question == 3:
        query = ("select Duration, Aspect_Ratio, Release_Year, Budget, Gross_Revanue, Avg_Rating, Votes, IMDB_Score "
                 "from IMDB.SHOW natural join IMDB.EARNINGS natural join IMDB.SCORE where Duration is not null")
    elif question == 4:
        query = ("SELECT Show_Id, Title, Genre, IMDB_Score "
                 "FROM IMDB.SHOW NATURAL JOIN IMDB.GENRE NATURAL JOIN IMDB.SCORE")
    elif question == 5:
        query = ("SELECT Show_Id, Title, Release_Year, Gross_Revanue, Genre, IMDB_Score, Name "
                 "FROM IMDB.SHOW NATURAL JOIN IMDB.EARNINGS NATURAL JOIN IMDB.GENRE NATURAL JOIN "
                 "IMDB.SCORE NATURAL JOIN IMDB.WORKED_ON NATURAL JOIN IMDB.PERSON "
                 "WHERE Role = 'Actor'")

    cursor.execute(query)
    data = cursor.fetchall()
    return data

#generate features and labels
def make_dataset(question, data):
    features = []

    if question == 1:
        model = ExtraTreesClassifier()
        labels = ["Gross_Revanue", "Budget", "Duration", "Aspect_Ratio", "Release_Year", "Votes", "IMDB_Score"]

        #importance relative to gross revenue
        X = np.array([row[1:] for row in data])
        Y = [row[0] for row in data]

        model.fit(X,Y)
        importances = model.feature_importances_

        result = dict(zip(labels[1:], importances))

        print("For gross revenue")
        for label,value in sorted(result.items(), key=operator.itemgetter(1)):
            print(label, value)

        #importances relative to score
        X = np.array([row[:6] for row in data])
        Y = [row[6] for row in data]

        model.fit(X,Y)
        importances = model.feature_importances_

        result = dict(zip(labels[:6], importances))

        print("\nFor IMDB score")
        for label,value in sorted(result.items(), key=operator.itemgetter(1)):
            print(label, value)
    elif question == 2:
        X = np.array(data)
    elif question == 3:
        X = np.array(normalize(data)) #Normalize data and store in a numpy array (matrix)

        pca = PCA(n_components=2)
        X_r = pca.fit(X).transform(X) #Perform PCA on data matrix and store result

        components = [pca.components_[0], pca.components_[1]] #get principal components
        print(pca.explained_variance_ratio_) #Print eigenvals to show variance of each column

        for x in X_r:
            plt.scatter(components[0][0]*x[0],
                components[0][1]*x[0], color="r")
            plt.scatter(components[1][0]*x[1],
                components[1][1]*x[1], color="b")

        plt.show()

        for x in X_r:
            plt.scatter(x[0], x[1], color="g")

        plt.show()
    elif question == 4:
        X = np.array(data)
    elif question == 5:
        X = np.array(data)

    return features

#analyze and return auc
def cross_validation(name, features, labels, num_folds):
    if name.lower() == 'knn':
        data = cross_val_score(KNeighborsClassifier(),
            features, labels, cv=num_folds, scoring='roc_auc')
    elif name.lower() == 'gnb':
        data = cross_val_score(GaussianNB(), features.toarray(),
            labels, cv=num_folds, scoring='roc_auc')
    elif name.lower() == 'tree':
        data = cross_val_score(DecisionTreeClassifier(), features.toarray(),
            labels, cv=num_folds, scoring='roc_auc')
    elif name.lower() == 'svc':
        data = cross_val_score(LinearSVC(), features.toarray(),
            labels, cv=num_folds, scoring='roc_auc')
    elif name.lower() == 'mlp':
        data = cross_val_score(MLPClassifier(), features.toarray(),
            labels, cv=num_folds, scoring='roc_auc')
    auc = np.mean(data)
    return auc

if __name__ == "__main__":
    question = int(sys.argv[1]) #e.g. 1
    # name = sys.argv[2] #e.g. knn
    # num_folds = int(sys.argv[3]) #e.g. 5

    # Initialize connection to database
    db = dbc.connect(port = 3306,
                     user = "root",
                     passwd = "password",
                     db = "IMDB")
    cursor = db.cursor()

    data = load_data(question, cursor)
    features = make_dataset(question, data)

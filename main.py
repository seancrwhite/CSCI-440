#authors: Tia Smith, Sean White
#description: Find answers to questions about films by querying IMDb database and performing analysis and visualization using machine learning techniques

import sys
import numpy as np
import operator
import heapq
import matplotlib.pyplot as plt
import mysql.connector as dbc
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

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
        query = ("SELECT Title, Description, Genre "
                 "FROM IMDB.SHOW NATURAL JOIN IMDB.GENRE "
                 "WHERE Description IS NOT NULL")
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
    if question == 1:
        model = ExtraTreesClassifier()
        labels = ["Gross_Revanue", "Budget", "Duration", "Aspect_Ratio", "Release_Year", "Votes", "IMDB_Score"]

        #importance relative to gross revenue
        X = np.array([row[1:] for row in data])
        Y = [row[0] for row in data]

        model.fit(X,Y)
        importances = model.feature_importances_

        y_pos = np.arange(len(labels[1:]))

        #Create bar chart
        plt.bar(y_pos, importances)
        plt.xticks(y_pos, labels[1:])
        plt.ylabel('Importance')
        plt.title('Feature Importance Relative to Gross Revenue')

        plt.show()

        #importances relative to score
        X = np.array([row[:6] for row in data])
        Y = [row[6] for row in data]

        model.fit(X,Y)
        importances = model.feature_importances_

        y_pos = np.arange(len(labels[:6]))

        #Create bar chart
        plt.bar(y_pos, importances)
        plt.xticks(y_pos, labels[:6])
        plt.ylabel('Importance')
        plt.title('Feature Importance Relative to IMDB Score')

        plt.show()
    elif question == 2:
        X = np.array(data)
        actor_d = []
        #add each uniqe actor to actor_d
        #for each occurence update count, add
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

        #Perform bag of words on the data
        count_vect = CountVectorizer(stop_words='english')

        X_t = count_vect.fit_transform([row[0] for row in X])
        freqs_t = sorted([(word, X_t.getcol(idx).sum()) for word, idx
            in count_vect.vocabulary_.items()], key = lambda x: -x[1])[:10]

        X_d = count_vect.fit_transform([row[1] for row in X])
        freqs_d = sorted([(word, X_d.getcol(idx).sum()) for word, idx
            in count_vect.vocabulary_.items()], key = lambda x: -x[1])[:10]

        #Visualize
        y_pos = range(len(freqs_t))

        plt.bar(y_pos, [val[1] for val in freqs_t], color='#226764')
        plt.xticks(y_pos, [val[0] for val in freqs_t])

        plt.show()

        plt.bar(y_pos, [val[1] for val in freqs_d], color='#AA6B39')
        plt.xticks(y_pos, [val[0] for val in freqs_d])

        plt.show()
    elif question == 5:
        X = np.array(data)
        # get actor data
        # calulate death age
        # get avg score, total revenue, amount of movies, total num votes
        # do multiple regression for death age
        # scatter plot of big name actors and when they will die

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
    make_dataset(question, data)

import numpy as np
import operator
import heapq
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet, BayesianRidge

class Modeler:
    def create_graph(self, data):
        curr_title = data[0][0]
        curr_cast = []
        relationship_dict = {}

        for row in data:
            if curr_title == row[0]:
                curr_cast.append(row[1])
            else:
                for actor in curr_cast:
                    if actor not in relationship_dict.keys():
                        relationship_dict.update({actor: []})

                    actor_relationships = relationship_dict[actor]

                    for coworker in curr_cast:
                        if coworker != actor and coworker not in actor_relationships:
                            actor_relationships.append(coworker)

                    relationship_dict.update({actor: actor_relationships})

                curr_cast = [row[1]]
                curr_title = row[0]

        return relationship_dict

    # Find important features in data using forests of trees
    def extract_feature_importance(self, data, targets):
        model = ExtraTreesClassifier()
        X = normalize(data)
        y = targets

        model.fit(X, y)
        importances = model.feature_importances_

        return importances

    # Perform principal component analysis on data and return result
    def pca(self, data, dim):
        pca = PCA(n_components=dim)

        X = pca.fit(data).transform(data)

        return X

    # Perform bag of words on the data and return ordered list of words by frequency
    def extract_word_freqs(self, data):
        count_vect = CountVectorizer(stop_words='english')

        X = count_vect.fit_transform(data)
        freqs = sorted([(word, X.getcol(idx).sum()) for word, idx
            in count_vect.vocabulary_.items()], key = lambda x: -x[1])[:10]

        return freqs

    def seperate_actors(self, data):
        # format actor data
        actors = {}
        for x in data:
            if x[0] in actors:
                actor_info = actors[x[0]]
                actor_info[2] += 1
                actor_info[3] += x[3]

                actors.update({x[0]: actor_info})
            else:
                actors.update({x[0]: [x[1], x[2], 1, x[3]]})

        # seperate living and deceased
        actors_l = {}
        actors_d = {}

        for name in actors:
            info = actors[name]
            if info[1] is None: # If death date null, actor is alive
                actors_l.update({name: [info[2], info[3], 0]})

            else:
                actors_d.update({name: [info[2], info[3], info[1] - info[0]]})

        return actors_l, actors_d

    def eval_regression_models(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        scores = []

        linear_regression = LinearRegression()
        linear_regression.fit(X_train, y_train)
        scores.append('Linear Regression: {}'.format(linear_regression.score(X_test, y_test)))

        logistic_regression = LogisticRegression()
        logistic_regression.fit(X_train, y_train)
        scores.append('Logistic Regression: {}'.format(logistic_regression.score(X_test, y_test)))

        ridge = Ridge()
        ridge.fit(X_train, y_train)
        scores.append('Ridge: {}'.format(ridge.score(X_test, y_test)))

        lasso = Lasso()
        lasso.fit(X_train, y_train)
        scores.append('Lasso: {}'.format(lasso.score(X_test, y_test)))

        elastic_net = ElasticNet()
        elastic_net.fit(X_train, y_train)
        scores.append('Elastic Net: {}'.format(elastic_net.score(X_test, y_test)))

        bayes = BayesianRidge()
        bayes.fit(X_train, y_train)
        scores.append('Bayesian: {}'.format(bayes.score(X_test, y_test)))

        return scores

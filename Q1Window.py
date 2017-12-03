import numpy as np
from PyQt5.QtWidgets import QWidget, QPushButton
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from DataLoader import DataLoader
from Modeler import Modeler

class Q1Window:
    def __init__(self):
        self.initUI()

    def initUI(self):
        self.w = QWidget()
        self.w.resize(600, 300)
        self.w.move(350, 350)
        self.w.setWindowTitle('Question 1')

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.axes = self.figure.add_subplot(111)

        loader = DataLoader()
        data = loader.fetch_data(1)

        btn_f1 = QPushButton('Fig 1: Revenue', self.w)
        btn_f2 = QPushButton('Fig 2: Score', self.w)

        btn_f1.move(5, 5)
        btn_f2.move(125, 5)

        btn_f1.clicked.connect(self.on_click_F1)
        btn_f2.clicked.connect(self.on_click_F2)

        self.importances_r = self.get_revenue_importances(data)
        self.importances_s = self.get_score_importances(data)

        self.w.show()

    def on_click_F1(self):
        self.canvas.axes.clear()

        labels = ["Budget", "Duration","Aspect Ratio",
            "Release Year", "Votes", "IMDB Score"]
        x_pos = np.arange(len(self.importances_r))

        ax = self.figure.add_subplot(111)

        ax.bar(x_pos, self.importances_r)

        ax.set_ylabel('Importance')
        ax.set_title('Feature Importance Relative to Revenue')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels)

        self.canvas.draw()
        self.canvas.show()

    def on_click_F2(self):
        self.canvas.axes.clear()
        
        labels = ["Gross Revenue", "Budget", "Duration",
            "Aspect Ratio", "Release Year", "Votes"]
        x_pos = np.arange(len(self.importances_s))

        ax = self.figure.add_subplot(111)

        ax.bar(x_pos, self.importances_s)

        ax.set_ylabel('Importance')
        ax.set_title('Feature Importance Relative to Score')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels)

        self.canvas.draw()
        self.canvas.show()

    #importance relative to gross revenue
    def get_revenue_importances(self, data):
        modeler = Modeler()

        X = np.array([row[1:] for row in data])
        Y = [row[0] for row in data]

        importances = modeler.extract_feature_importance(X, Y)
        return importances

    #importance relative to score
    def get_score_importances(self, data):
        modeler = Modeler()

        X = np.array([row[:6] for row in data])
        Y = [row[6] for row in data]

        importances = modeler.extract_feature_importance(X, Y)
        return importances

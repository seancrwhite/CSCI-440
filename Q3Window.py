import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from DataLoader import DataLoader
from Modeler import Modeler
from PyQt5.QtWidgets import QWidget, QPushButton
from sklearn.preprocessing import normalize

class Q3Window:
    def __init__(self):
        self.initUI()

    def initUI(self):
        self.w = QWidget()
        self.w.resize(600, 300)
        self.w.move(350, 350)
        self.w.setWindowTitle('Question 3')

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.axes = self.figure.add_subplot(111)

        loader = DataLoader()
        data = loader.fetch_data(3)
        data_n = normalize(data)
        modeler = Modeler()

        btn_f1 = QPushButton('Fig 1.1: Spread', self.w)
        btn_f2 = QPushButton('Fig 1.2: Spread (Normalized)', self.w)

        btn_f1.move(5, 5)
        btn_f2.move(190, 5)

        btn_f1.clicked.connect(self.on_click_F1)
        btn_f2.clicked.connect(self.on_click_F2)

        self.transformed_data = modeler.pca(data, 2)
        self.transformed_data_n = modeler.pca(data_n, 2)

        self.w.show()

    def on_click_F1(self):
        self.canvas.axes.clear()

        ax = self.figure.add_subplot(111)

        self.plot(False, ax)

        self.canvas.draw()
        self.canvas.show()

    def on_click_F2(self):
        self.canvas.axes.clear()
        ax = self.figure.add_subplot(111)

        self.plot(True, ax)

        self.canvas.draw()
        self.canvas.show()

    def plot(self, normalize_data, ax):
        if not normalize_data:
            ax.scatter(self.transformed_data[:,0],self.transformed_data[:,1],
                c=['#AA4F39', '#256F5C', '#3C8D2F', '#111111'])
        else:
            ax.scatter(self.transformed_data_n[:,0],self.transformed_data_n[:,1],
                c=['#AA4F39', '#256F5C', '#3C8D2F', '#111111'])

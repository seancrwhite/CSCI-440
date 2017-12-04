import numpy as np
from PyQt5.QtWidgets import QWidget, QPushButton, QLabel, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from DataLoader import DataLoader
from Modeler import Modeler

class Q5Window:
    def __init__(self):
        self.initUI()

    def initUI(self):
        self.w = QWidget()
        self.w.resize(600, 300)
        self.w.move(350, 350)
        self.w.setWindowTitle('Question 5')

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.axes = self.figure.add_subplot(111)

        loader = DataLoader()
        data = loader.fetch_data(5)
        modeler = Modeler()

        actors_l, actors_d = modeler.seperate_actors(data)

        vals_d = np.array(list(actors_d.values()))
        X_d = vals_d[:,:2]
        y_d = vals_d[:,2]

        scores = modeler.eval_regression_models(X_d, y_d)

        l1 = QLabel()
        l2 = QLabel()
        l3 = QLabel()
        l4 = QLabel()
        l5 = QLabel()

        l1.setText(scores[0])
        l2.setText(scores[1])
        l3.setText(scores[2])
        l4.setText(scores[3])
        l5.setText(scores[4])

        vbox = QVBoxLayout()
        vbox.addWidget(l1)
        vbox.addWidget(l2)
        vbox.addWidget(l3)
        vbox.addWidget(l4)
        vbox.addWidget(l5)

        self.w.setLayout(vbox)
        self.w.show()

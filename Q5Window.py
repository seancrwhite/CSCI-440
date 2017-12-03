import numpy as np
from PyQt5.QtWidgets import QWidget, QPushButton
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

        modeler.eval_regression_models(X_d, y_d)

        self.w.show()

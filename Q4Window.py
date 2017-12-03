import numpy as np
from PyQt5.QtWidgets import QWidget, QPushButton
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from DataLoader import DataLoader
from Modeler import Modeler

class Q4Window:
    def __init__(self):
        self.initUI()

    def initUI(self):
        self.w = QWidget()
        self.w.resize(600, 300)
        self.w.move(350, 350)
        self.w.setWindowTitle('Question 4')

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.axes = self.figure.add_subplot(111)

        loader = DataLoader()
        data = np.array(loader.fetch_data(4))
        modeler = Modeler()

        btn_f1 = QPushButton('Fig 1: Title', self.w)
        btn_f2 = QPushButton('Fig 2: Description', self.w)

        btn_f1.move(5, 5)
        btn_f2.move(125, 5)

        btn_f1.clicked.connect(self.on_click_F1)
        btn_f2.clicked.connect(self.on_click_F2)

        self.freqs_t = np.array(modeler.extract_word_freqs(data[:,0]))
        self.freqs_d = np.array(modeler.extract_word_freqs(data[:,1]))

        print(self.freqs_t)

        self.w.show()

    def on_click_F1(self):
        self.canvas.axes.clear()

        x_pos = np.arange(len(self.freqs_t))

        ax = self.figure.add_subplot(111)

        ax.bar(x_pos, self.freqs_t[:,1])

        ax.set_ylabel('Frequency')
        ax.set_title('Most Common Words In Titles')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(self.freqs_t[:,0])

        self.canvas.draw()
        self.canvas.show()

    def on_click_F2(self):
        self.canvas.axes.clear()

        x_pos = np.arange(len(self.freqs_d))

        ax = self.figure.add_subplot(111)

        ax.bar(x_pos, self.freqs_d[:,1])

        ax.set_ylabel('Frequency')
        ax.set_title('Most Common Words In Descriptions')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(self.freqs_d[:,0])

        self.canvas.draw()
        self.canvas.show()

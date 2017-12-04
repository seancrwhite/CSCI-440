import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from DataLoader import DataLoader
from Modeler import Modeler
from PyQt5.QtWidgets import QWidget, QPushButton
from sklearn.preprocessing import normalize
import networkx as nx

class Q2Window:
    def __init__(self):
        self.initUI()

    def initUI(self):
        self.w = QWidget()
        self.w.resize(600, 300)
        self.w.move(350, 350)
        self.w.setWindowTitle('Question 2')

        loader = DataLoader()
        data = loader.fetch_data(2)
        modeler = Modeler()

        relationship_dict = modeler.create_graph(data)
        relationship_graph = nx.Graph(relationship_dict)

        nx.draw(relationship_graph, node_size=2)
        plt.show()

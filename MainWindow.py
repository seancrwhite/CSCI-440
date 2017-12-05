from PyQt5.QtWidgets import QWidget, QPushButton
from Q1Window import Q1Window
from Q2Window import Q2Window
from Q3Window import Q3Window
from Q4Window import Q4Window
from Q5Window import Q5Window

class MainWindow:
    def __init__(self):
        self.initUI() # Initialize window on startup

    # Create application window
    def initUI(self):
        self.w = QWidget()
        self.w.resize(600, 300)
        self.w.move(300, 300)
        self.w.setWindowTitle('Data Science Final Project')

        # Initialize buttons
        btn_q1 = QPushButton('Q1: Feature Importance', self.w)
        btn_q2 = QPushButton('Q2: Relationship Graph', self.w)
        btn_q3 = QPushButton('Q3: Principal Component Analysis', self.w)
        btn_q4 = QPushButton('Q4: Buzzword Analysis', self.w)
        btn_q5 = QPushButton('Q5: Death Age Predictor', self.w)

        # Place them on screen
        btn_q1.move(50, 50)
        btn_q2.move(50, 80)
        btn_q3.move(50, 110)
        btn_q4.move(50, 140)
        btn_q5.move(50, 170)

        # Add functionality
        btn_q1.clicked.connect(self.on_click_Q1)
        btn_q2.clicked.connect(self.on_click_Q2)
        btn_q3.clicked.connect(self.on_click_Q3)
        btn_q4.clicked.connect(self.on_click_Q4)
        btn_q5.clicked.connect(self.on_click_Q5)

        self.w.show()

    # Open window 1 on button click
    def on_click_Q1(self):
        self.q1_window = Q1Window()

    # Open window 2 on button click
    def on_click_Q2(self):
        self.q2_window = Q2Window()

    # Open window 3 on button click
    def on_click_Q3(self):
        self.q3_window = Q3Window()

    # Open window 4 on button click
    def on_click_Q4(self):
        self.q4_window = Q4Window()

    # Open window 5 on button click
    def on_click_Q5(self):
        self.q5_window = Q5Window()

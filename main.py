#authors: Tia Smith, Sean White
#description: Find answers to questions about films by querying IMDb database and performing analysis and visualization using machine learning techniques

import sys
from DataLoader import DataLoader
from MainWindow import MainWindow
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

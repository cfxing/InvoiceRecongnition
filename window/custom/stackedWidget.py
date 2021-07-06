from window.custom.tableWidget import *
from window.config import tables


class StackedWidget(QStackedWidget):
    def __init__(self, parent):
        super().__init__(parent=parent)
        for table in tables:
            self.addWidget(table(parent=parent))
        self.setMinimumWidth(200)
        self.setMinimumHeight(300)

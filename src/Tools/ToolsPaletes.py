from PyQt6.QtWidgets import QToolBox, QWidget, QVBoxLayout, QPushButton, QLabel

from Tools.ColorDials import ColorDials
from Tools.CVtools import CVtools


class ToolsPaletes(QWidget):
    def __init__(self, topWidget, engine):
        super().__init__(topWidget)
        self.engine = engine

        layout = QVBoxLayout()
        tlbx = QToolBox()

        tlbx.addItem(ColorDials(self, engine), 'Точечные операции')
        tlbx.addItem(CVtools(self, engine), 'Морфологические операции')

        apply_thinning = QPushButton('Построить скелет')
        apply_thinning.clicked.connect(self.engine.thinning)

        apply_postproccessing = QPushButton('Обработать скелет')
        apply_postproccessing.clicked.connect(self.engine.postproccessing)

        self.class_detected_label = QLabel('Класс: None')
        

        layout.addWidget(tlbx)
        layout.addWidget(apply_thinning)
        layout.addWidget(apply_postproccessing)
        layout.addWidget(self.class_detected_label)

        self.setMinimumWidth(400)
        self.setMinimumHeight(550)
        self.setLayout(layout)

        self.engine.add_refresh_action(self.check_on_class_detected)
        self.engine.add_reset_action(self.check_on_class_detected)

    def check_on_class_detected(self):
        self.class_detected_label.setText(f'Класс: {self.engine.get_class_detected()}')

from PyQt6.QtWidgets import QWidget, QFormLayout, QPushButton, QHBoxLayout, QVBoxLayout, QSlider, QCheckBox, QLabel
from PyQt6.QtCore import QSize
from PyQt6.QtCore import Qt

CLICKED_BUTTON_STYLE = '''
    background-color: #4e80ee;
    border-radius: 4px; 
    padding: 14px; 
    padding-top: 2px; 
    padding-bottom: 2px; 
    margin: 1px;
    margin-top: 1px;
    margin-bottom: 1px;
'''


class CVtools(QWidget):
    class __morph_tool(QWidget):
        def __init__(self, topWidget, func):
            super().__init__(topWidget)
            self.func = func

            # define layout
            self.layout = QHBoxLayout()

            # add kernel size define-slider
            self.slider = QSlider(Qt.Orientation.Horizontal)
            self.slider.setRange(0, 30)
            self.slider.setValue(0)
            self.slider.setFixedSize(QSize(200, 30))
            self.slider.valueChanged.connect(self.func)
            self.layout.addWidget(self.slider)

            self.setLayout(self.layout)
        
        def clear_morph(self):
            self.slider.setValue(0)

    def __init__(self, topModel, engine):
        super().__init__(topModel)
        self.engine = engine

        self.opening_before_tool  = self.__morph_tool(self, self.engine.opening_before_change)
        self.erosion_before_tool  = self.__morph_tool(self, self.engine.erosion_before_change)
        self.closing_tool  = self.__morph_tool(self, self.engine.closing_change)
        self.dilation_tool = self.__morph_tool(self, self.engine.dilation_change)
        self.opening_after_tool  = self.__morph_tool(self, self.engine.opening_after_change)
        self.erosion_after_tool  = self.__morph_tool(self, self.engine.erosion_after_change)

        self.layout = QFormLayout()
        self.layout.addRow('Opening before: ', self.opening_before_tool)
        self.layout.addRow('Erosion before: ', self.erosion_before_tool)
        self.layout.addRow('Closing: ', self.closing_tool)
        self.layout.addRow('Dilation: ', self.dilation_tool)
        self.layout.addRow('Opening after: ', self.opening_after_tool)
        self.layout.addRow('Erosion after: ', self.erosion_after_tool)

        self.engine.add_reset_action(self.clear_styles)
        self.setLayout(self.layout)

    def clear_styles(self):
        self.opening_before_tool.clear_morph()
        self.erosion_before_tool.clear_morph()
        self.closing_tool.clear_morph()
        self.dilation_tool.clear_morph()
        self.opening_after_tool.clear_morph()
        self.erosion_after_tool.clear_morph()


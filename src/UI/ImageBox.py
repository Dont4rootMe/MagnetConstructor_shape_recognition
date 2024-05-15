from PyQt6.QtWidgets import QLabel
from PyQt6.QtCore import QSize

class ImageBoxAbstract(QLabel):
    def __init__(self, topWidget, engine):
        super().__init__(topWidget)
        self.engine = engine
        self.engine.add_refresh_action(self.actionOnRefreshReset())
        self.engine.add_reset_action(self.actionOnRefreshReset())

        self.setScaledContents(True)
        self.setFixedSize(QSize(500, 500))
        # self.setMaximumSize(QSize(650, 300))
        # self.setMinimumSize(QSize(300, 300))


        self.index = []

class OriginalImage(ImageBoxAbstract):
    def __init__(self, topWidget, engine):
        super().__init__(topWidget, engine)

        self.mousePressEvent = self.getPos

    def update_pic(self):
        if self.engine.original_pixmap_exist():
            self.setPixmap(self.engine.get_original_pixmap())

    def getPos(self , event):
        import numpy as np
        x = event.pos().y()
        y = event.pos().x()

        true_shape = self.rect().getRect()
        shape = np.array(self.engine.img).shape
        # print((x, y), (int(y * shape[1] / (true_shape[2] - 1)), int(x * shape[0] / (true_shape[3] - 1))) - 10)
        # print(shape, true_shape, (x, y), (int(x * shape[0] / true_shape[3]), int(y * shape[1] / true_shape[2])))
        self.engine.temp_set_range((int(y * shape[1] / true_shape[2]), int(x * shape[0] / true_shape[3])))
        # colors = np.array(self.engine.img).astype(float)[x, y]

        # self.index.append(colors)
        # if len(self.index) > 500:
        #     np.array(self.index).dump('negative1.npy')


        

    def actionOnRefreshReset(self):
        return self.update_pic
    
class ModifiedImage(ImageBoxAbstract):
    def __init__(self, topWidget, engine):
        super().__init__(topWidget, engine)

    def update_pic(self):
        if self.engine.modified_pixmap_exist():
            self.setPixmap(self.engine.get_modified_pixmap())

    def actionOnRefreshReset(self):
        return self.update_pic
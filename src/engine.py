from PyQt6.QtGui import QPixmap
from PIL import Image, ImageEnhance, ImageQt

from PIL import Image, ImageEnhance, ImageQt
from skimage.morphology import skeletonize
from skimage import data
import numpy as np
import cv2 as cv
from itertools import combinations

from models import get_graph

class Engine:
    def __refresher__(clear_thinning=True):
        def __decorator__(func):
            def inner(self, *args, **kwargs):
                func(self, *args, **kwargs)
                if clear_thinning:
                    self.actions['thinning'] = False
                    self.actions['postproccess'] = False
                self.refresh()
            return inner
        
        return __decorator__

    def __whipe_dict(self):
        self.actions = {
            'brightness': False,
            'contrast': False,
            'sharpness': False,
            'saturation': False,
            'color_enhancements': [1.0, 1.0, 1.0],
            'binarize': 128,
            'invert_binarize': False,
            'thinning': False,
            'postproccess': False,
            'erosion': None,
            'dilation': None,
            'opening': None,
            'closing': None,
        }

        self.min_contour_len = 500
        self.contours_detected = None
        self.triangle_contours = None
        self.convexes = None

    def __init__(self):
        self.kernel = [] #    !!!!!!!!!!!!!!!!!!!
        self.index_kernel = 0


        self.img = None
        self.modified = None

        self.refresh_actions = []
        self.reset_actions = []

        self.__whipe_dict()

    def refresh(self):
        for act in self.refresh_actions:
            act()
    
    def reset(self):
        for act in self.reset_actions:
            act()
        
        self.__whipe_dict()

    def add_refresh_action(self, action):
        self.refresh_actions.append(action)

    def add_reset_action(self, action):
        self.reset_actions.append(action)

    def upload_picture(self, path):
        self.img = Image.open(path).copy()
        self.modified = Image.open(path).copy()
        self.reset()

    def original_pixmap_exist(self):
        return self.img is not None
    
    def modified_pixmap_exist(self):
        return self.modified is not None

    def get_original_pixmap(self):
        temp = self.img

        points = np.array([
            # positive
            [ 70.2, 100.9,  89.2],
            [141.8, 167.9, 166.6],
            [86.9, 118.7, 117.5],
            [102.5, 137.9, 132.6],
            [156.6, 181.0, 177.1],
            [88.1, 83.4, 79.9],
            [56.1, 46.3, 39.8],
            [171.7, 168.4, 169.4],
            [121.0,  146.1, 146.0],
            [41.9, 52.6, 37.1],
            [55.75, 62.08, 42.91],
            [42.9, 51.1, 36.62],
            [44.1, 39.0, 37.8],
            [47.3, 42.2, 38.7],
            [21.2, 15.9, 13.3],
            [122.17142857, 151.25714286, 150.64761905],
            [132.34782609, 155.74782609, 145.36521739],
            [109.27878788, 137.62424242, 126.45454545],
            [73.09567497, 92.88073394, 83.81913499],
            [35.77922078, 36.46753247, 34.80519481],

            # negative
            [146.6,  148.6, 143.7],
            [52.41, 59.3, 69.3],
            [138.99157895, 111.16631579,  97.06105263],
            [128.91331269, 114.12631579,  87.72136223],
            [130.9, 127.8, 119.13],
            [65.53623188, 59.98550725, 57.91666667],
            [99.71186441, 101.05932203, 102.52542373],
            [150.,          72.98511905,  46.64880952],
            [85.58898305, 70.14830508, 43.35911017],
            [152.10536913, 149.09530201, 143.78456376],
            [119.42290289, 118.34528368, 116.58475461],
            [152.27350427, 143.55718356, 136.43589744],
            [110.0413551,  102.78589959,  98.01693881],
            [158.50596462, 157.93377211, 152.99053887],
            [91.93076923, 81.32307692, 54.7],
            [54.41782178, 40.52079208, 35.72475248],
        ])

        # temp = np.array(temp)
        # temp = cv.blur(temp, (9,9), 100)
        # if len(self.kernel) >= 4:
        #     import os
        #     file = f'./data/negative{len(os.listdir("./data"))}.npy'

        #     mask = cv.fillPoly(np.zeros((temp.shape[0], temp.shape[1])), [np.array(self.kernel)], 255) > 0
        #     data = temp[mask].reshape((-1, 3))
            # for p in points:
            #     delta = np.linalg.norm(data[:, :3] - p[None, :], axis=-1)
            #     data = np.concatenate([data, delta[:, None]], axis=-1)
        #     print(data.shape)

        #     data.dump(file)
        #     temp[mask] = 0 
        #     test = np.load(file, allow_pickle=True)
        #     temp[mask] = test[:, :3] / 2
        #     self.kernel = []

        #     # temp[mask].reshape((-1, 3)).dump(file)
        #     # temp[mask] = 0 
        #     # test = np.load(file, allow_pickle=True)
        #     # temp[mask] = test / 2
        #     self.kernel = []
        # temp = Image.fromarray(temp.astype(np.uint8))
        # return QPixmap.fromImage(ImageQt.ImageQt(temp).copy())


#         Matrix = np.array(temp)
#         Matrix = cv.blur(Matrix, (9,9))
#         Matrix = cv.resize(Matrix, (0,0), fx=0.4, fy=0.4) 

#         data = Matrix.reshape((-1, 3))
#         for p in points:
#             delta = np.linalg.norm(data[:, :3] - p[None, :], axis=-1)
#             data = np.concatenate([data, delta[:, None]], axis=-1)

#         import torch
#         model = torch.nn.Sequential(
#     torch.nn.Linear(39, 60),
#     torch.nn.ReLU(),
#     torch.nn.BatchNorm1d(60),
#     torch.nn.Linear(60, 60),
#     torch.nn.ReLU(),
#     torch.nn.BatchNorm1d(60),
#     torch.nn.Linear(60, 60),
#     torch.nn.ReLU(),
#     torch.nn.BatchNorm1d(60),
#     torch.nn.Linear(60, 60),
#     torch.nn.ReLU(),
#     torch.nn.BatchNorm1d(60),
#     torch.nn.Linear(60, 2)
# )
#         model.load_state_dict(torch.load('segmentation_model.pt'))
#         mask = (model(torch.tensor(data, dtype=torch.float32))[:, 1] >= 0).numpy().astype(np.uint8).reshape((Matrix.shape[0], Matrix.shape[1])) * 255
#         temp = Image.fromarray(mask.astype(np.uint8))
        # apply pixel-wise operations
        if self.actions['brightness']:
            enhancer = ImageEnhance.Brightness(temp)
            temp = enhancer.enhance(self.actions['brightness'])
        if self.actions['contrast']:
            enhancer = ImageEnhance.Contrast(temp)
            temp = enhancer.enhance(self.actions['contrast'])
        if self.actions['sharpness']:
            enhancer = ImageEnhance.Sharpness(temp)
            temp = enhancer.enhance(self.actions['sharpness'])
        if self.actions['saturation']:
            enhancer = ImageEnhance.Color(temp)
            temp = enhancer.enhance(self.actions['saturation'])

        # apply color enhancements methods
        Matrix = np.array(temp).astype(float)
        Matrix[..., 0] *= self.actions['color_enhancements'][0]
        Matrix[..., 1] *= self.actions['color_enhancements'][1]
        Matrix[..., 2] *= self.actions['color_enhancements'][2]
        temp = Image.fromarray(Matrix.astype(np.uint8))


        return QPixmap.fromImage(ImageQt.ImageQt(temp).copy())
        
    
    def get_modified_pixmap(self):
        temp = self.img

        # apply pixel-wise operations
        if self.actions['brightness']:
            enhancer = ImageEnhance.Brightness(temp)
            temp = enhancer.enhance(self.actions['brightness'])
        if self.actions['contrast']:
            enhancer = ImageEnhance.Contrast(temp)
            temp = enhancer.enhance(self.actions['contrast'])
        if self.actions['sharpness']:
            enhancer = ImageEnhance.Sharpness(temp)
            temp = enhancer.enhance(self.actions['sharpness'])
        if self.actions['saturation']:
            enhancer = ImageEnhance.Color(temp)
            temp = enhancer.enhance(self.actions['saturation'])

        # apply color enhancements methods
        Matrix = np.array(temp).astype(float)
        Matrix[..., 0] *= self.actions['color_enhancements'][0]
        Matrix[..., 1] *= self.actions['color_enhancements'][1]
        Matrix[..., 2] *= self.actions['color_enhancements'][2]
        temp = Image.fromarray(Matrix.astype(np.uint8))

        # binariaze and invert binarization
        Matrix = np.array(temp).astype(np.float32)
        Matrix = cv.cvtColor(Matrix, cv.COLOR_BGR2GRAY)
        ret, temp = cv.threshold(
            Matrix, self.actions['binarize'], 255, cv.THRESH_BINARY)
        # if it is neccessery invert colors
        if self.actions['invert_binarize']:
            temp = 255 - temp
        temp = Image.fromarray(temp.astype(np.uint8))


        # applying porphological operations
        if self.actions['erosion'] is not None:
            Matrix = np.array(temp).astype(float)
            temp = cv.erode(Matrix, self.actions['erosion'], iterations=1)
            temp = Image.fromarray(temp.astype(np.uint8))

        if self.actions['opening'] is not None:
            Matrix = np.array(temp).astype(float)
            temp = cv.morphologyEx(
                Matrix, cv.MORPH_OPEN, self.actions['opening'])
            temp = Image.fromarray(temp.astype(np.uint8))

        if self.actions['closing'] is not None:
            Matrix = np.array(temp).astype(float)
            temp = cv.morphologyEx(
                Matrix, cv.MORPH_CLOSE, self.actions['closing'])
            temp = Image.fromarray(temp.astype(np.uint8))

        if self.actions['dilation'] is not None:
            Matrix = np.array(temp).astype(float)
            temp = cv.dilate(Matrix, self.actions['dilation'], iterations=1)
            temp = Image.fromarray(temp.astype(np.uint8))


        if self.actions['thinning']:
            Matrix = np.array(temp).astype(np.uint8)
            temp = cv.ximgproc.thinning(Matrix, thinningType=cv.ximgproc.THINNING_GUOHALL)
            temp = Image.fromarray(temp.astype(np.uint8))

        if self.actions['postproccess']:
            Matrix = np.array(temp).astype(np.uint8)
            result_canvas, connectivity = get_graph(Matrix.copy())

            temp = Image.fromarray(result_canvas.astype(np.uint8) + Matrix.astype(np.uint8))

        self.modified = temp.copy()
        return QPixmap.fromImage(ImageQt.ImageQt(temp).copy())


    @__refresher__(True)
    def change_brightness(self, brightness):
        k = (brightness - 127) / 128 + 1.0
        self.actions['brightness'] = k

    @__refresher__(True)
    def change_contrast(self, contrast):
        k = (contrast - 127) / 128 + 1.0
        self.actions['contrast'] = k

    @__refresher__(True)
    def change_sharpness(self, sharpness):
        k = ((sharpness - 127) / 128) * 8 + 1.0
        self.actions['sharpness'] = k

    @__refresher__(True)
    def change_saturation(self, saturation):
        k = ((saturation - 127) / 128) + 1.0
        self.actions['saturation'] = k

    @__refresher__(True)
    def enhance_red(self, red):
        k = (red - 300) / 300 + 1.0
        self.actions['color_enhancements'][0] = k

    @__refresher__(True)
    def enhance_green(self, green):
        k = (green - 300) / 300 + 1.0
        self.actions['color_enhancements'][1] = k

    @__refresher__(True)
    def enhance_blue(self, blue):
        k = (blue - 300) / 300 + 1.0
        self.actions['color_enhancements'][2] = k

    @__refresher__(True)
    def binarize(self, bin):
        self.actions['binarize'] = bin

    @__refresher__(True)
    def invert_binarize(self, val):
        self.actions['invert_binarize'] = val

    @__refresher__(True)
    def erosion_change(self, erosion):
        if erosion == 0:
            self.actions['erosion'] = None
            return
        kernel = np.ones((erosion, erosion), np.uint8)
        self.actions['erosion'] = kernel

    @__refresher__(True)
    def dilation_change(self, dilation):
        if dilation == 0:
            self.actions['dilation'] = None
            return
        kernel = np.ones((dilation, dilation), np.uint8)
        self.actions['dilation'] = kernel

    @__refresher__(True)
    def opening_change(self, opening):
        if opening == 0:
            self.actions['opening'] = None
            return
        kernel = np.ones((opening, opening), np.uint8)
        self.actions['opening'] = kernel

    @__refresher__(True)
    def closing_change(self, closing):
        if closing == 0:
            self.actions['closing'] = None
            return
        kernel = np.ones((closing, closing), np.uint8)
        self.actions['closing'] = kernel

    
    @__refresher__(False)
    def thinning(self, val):
        self.actions['thinning'] = True

    @__refresher__(False)
    def postproccessing(self, val):
        self.actions['postproccess'] = True


    @__refresher__(True)
    def temp_set_range(self, xy):
        self.kernel.append(xy)





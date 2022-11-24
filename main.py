import threading
import cv2
from PySide2.QtUiTools import QUiLoader
from PySide2.QtWidgets import QApplication

from utils import file_taking,show_img
from trainer import data_train
from detect import *
class Stats():

    def __init__(self):

        #ui读取
        self.ui = QUiLoader().load('ui/main.ui')
        self.video = threading.Thread(target=self.video_show)
        #设置按钮函数
        self.ui.button_train.clicked.connect(self.train)
        self.ui.button_file.clicked.connect(self.file_detect)
        self.ui.button_video.clicked.connect(self.video_show)

        self.video = threading.Thread(target=self.video_detect)
        self.cap = cv2.VideoCapture(0)



    def train(self):
        data_train(self)

    def file_detect(self):
        img = file_taking(self)
        img = detect_img(img)
        show_img(self,img)

    def video_detect(self):
        detect_video(self)

    def video_show(self):
        self.video.start()



if __name__ == '__main__':
    app = QApplication([])
    stats = Stats()
    stats.ui.show()
    app.exec_()







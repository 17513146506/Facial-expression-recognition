from utils import data_prepare
import cv2
import numpy as np

def data_train(self):

    faces, labels = data_prepare()
    print("Total faces: ", len(faces))
    print("Total labels: ", len(labels))

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    #应用数据，进行训练
    face_recognizer.train(faces, np.array(labels))
    face_recognizer.write("weight/trainer.yml")
    self.ui.edit.setText("训练完成")




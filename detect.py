import cv2
from utils import draw_text,draw_rectangle,get_region
from PySide2.QtGui import QPixmap, QImage
from PySide2.QtWidgets import QSizePolicy

def detect_img(img):
    img_a = cv2.imread(img)
    img_a = img_a.copy()
    a = ["happy","sad"]
    recogizer = cv2.face.LBPHFaceRecognizer_create()  # 加载数据训练文件
    recogizer.read("weight/trainer.yml")

    face,rec = get_region(img)#得到需要检测图片的面部部分数据
    label,conf = recogizer.predict(face)

    text = a[label]
    if conf > 80:

        draw_text(img_a, text, rec[0], rec[1] - 2)
        draw_rectangle(img_a,rec)
    #cv2.imshow("2",img_a)

    return img_a



def detect_video(self):
    while self.cap.isOpened():
        success, frame = self.cap.read()
        # RGB转BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        #frame = detect_img(frame)
        img = frame.copy()
        a = ["happy", "sad"]
        recogizer = cv2.face.LBPHFaceRecognizer_create()  # 加载数据训练文件
        recogizer.read("weight/trainer.yml")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier("weight/lbpcascade_frontalface.xml")  # 加载人脸识别器
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=1)
        if len(faces) == 0:
            self.ui.edit.setText("未检测出")
        else:
            # 提取面部区域
            (x, y, w, h) = faces[0]
            # 返回人脸及其所在区域
            face,rec = gray[y:y + w, x:x + h], faces[0]
  # 得到需要检测图片的面部部分数据
            label, conf = recogizer.predict(face)

            text = a[label]
            draw_text(img, text, rec[0], rec[1] - 2)
            draw_rectangle(img, rec)

            img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            self.ui.label_video.setPixmap(QPixmap.fromImage(img))
            # 按比例填充
            self.ui.label_video.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
            self.ui.label_video.setScaledContents(True)








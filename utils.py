import cv2
import os

def get_region(img):
    gray = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    #gray = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2GRAY)
    face_cascade = cv2.CascadeClassifier("weight/lbpcascade_frontalface.xml")#加载人脸识别器
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=1)
    if len(faces) == 0:
        return None, None
    # 提取面部区域
    (x, y, w, h) = faces[0]
    # 返回人脸及其所在区域
    return gray[y:y + w, x:x + h], faces[0]


def data_prepare():
    dirs_train = os.listdir("img/img_train")
    faces = []
    labels = []
    for image_path in dirs_train:
        if image_path[0] == 'h':
            label = 0
        else:
            label = 1

        image = 'img/img_train/' + image_path

        #image = cv2.imread(image_path, 0)

        face, rect = get_region(image)
        if face is not None:
            faces.append(face)
            labels.append(label)


    return faces, labels


def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)



def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


def show_img(self,frame):
    from PySide2.QtGui import QPixmap, QImage
    from PySide2.QtWidgets import QSizePolicy

    #frame = cv2.imread(frame)
    img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
    self.ui.label_video.setPixmap(QPixmap.fromImage(img))
    # 按比例填充
    self.ui.label_video.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
    self.ui.label_video.setScaledContents(True)


def file_taking(self):
    from PySide2.QtWidgets import QFileDialog
    filePath, _ = QFileDialog.getOpenFileName(self.ui, "选择你要上传的图片", "./", "*.*")
    print(filePath)
    self.ui.edit.setText(filePath)
    #img = cv2.imread(filePath,0)
    return filePath



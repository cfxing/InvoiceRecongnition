import cv2

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import time


class GraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super(GraphicsView, self).__init__(parent=parent)
        self._zoom = 0
        self._empty = True
        self._photo = QGraphicsPixmapItem()
        self._scene = QGraphicsScene(self)
        self._scene.addItem(self._photo)
        self.setScene(self._scene)
        self.setAlignment(Qt.AlignCenter)  # 居中显示
        self.setDragMode(QGraphicsView.ScrollHandDrag)  # 设置拖动
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setMinimumSize(640, 480)

    def contextMenuEvent(self, event):
        if not self.has_photo():
            return
        menu = QMenu()
        save_action = QAction('另存为', self)
        save_action.triggered.connect(self.save_current)  # 传递额外值
        menu.addAction(save_action)
        menu.exec(QCursor.pos())

    def save_current(self):
        file_name = QFileDialog.getSaveFileName(self, '另存为', '../../test_result/mySave/'+str(time.time()), 'Image files(*.jpg *.gif *.png)')[0]
        print(file_name)
        if file_name:
            self._photo.pixmap().save(file_name)


    def get_image(self):
        if self.has_photo():
            return self._photo.pixmap().toImage()

    def has_photo(self):
        return not self._empty

    def change_image(self, img):
        self.update_image(img)
        self.fitInView()

    # 返回 pix 用于创建像素图元，  selt.item = QGraphicsPixmapItem(pix)
    def img_to_pixmap(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # bgr -> rgb
        h, w, c = img.shape  # 获取图片形状
        image = QImage(img, w, h, 3 * w, QImage.Format_RGB888)
        return QPixmap.fromImage(image)

    def update_image(self, img):
        self._empty = False
        self._photo.setPixmap(self.img_to_pixmap(img))

    def fitInView(self, scale=True):
        rect = QRectF(self._photo.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            if self.has_photo():
                unity = self.transform().mapRect(QRectF(0, 0, 1, 1))
                self.scale(1 / unity.width(), 1 / unity.height())
                viewrect = self.viewport().rect()
                scenerect = self.transform().mapRect(rect)
                factor = min(viewrect.width() / scenerect.width(),
                             viewrect.height() / scenerect.height())
                self.scale(factor, factor)
            self._zoom = 0

    def wheelEvent(self, event):
        if self.has_photo():
            if event.angleDelta().y() > 0:
                factor = 1.25
                self._zoom += 1
            else:
                factor = 0.8
                self._zoom -= 1
            if self._zoom > 0:
                self.scale(factor, factor)
            elif self._zoom == 0:
                self.fitInView()
            else:
                self._zoom = 0

    def select_button_clicked(self):
        file_name = QFileDialog.getOpenFileName(self, 'Open file', '../test_images', 'Image files(*.jpg *.gif *.png)')
        image_path = file_name[0]
        # if (file_name[0] == ""):
        #     QMessageBox.information(self, "提示", self.tr("没有选择图片文件！"))
        print(image_path)
        img = cv2.imread(image_path)  # 读取图像
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换图像通道
        x = img.shape[1]  # 获取图像大小
        y = img.shape[0]
        self.zoomscale = 1  # 图片放缩尺度
        frame = QImage(img, x, y, QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        self.item = QGraphicsPixmapItem(pix)  # 创建像素图元
        self.scene = QGraphicsScene()  # 创建场景
        self.scene.addItem(self.item)
        # self.graphicsView.setScene(self.scene)
        # self.graphicsView.show()
        self.setScene(self.scene)
        self.show()

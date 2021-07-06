import sys
import cv2
import numpy as np
from PyQt5.QtGui import *
from PyQt5.Qt import *
import shutil


from window.custom.listWidgets import FuncListWidget, UsedListWidget,ShowResWidget
from window.custom.graphicsView import GraphicsView
from window.custom.stackedWidget import StackedWidget
import os
import random
import recognition
from cut_e_invoice import cut_image_e
from cut_p_invoice import cut_img_m
import re

res_dir = r'D:\Devsoft\PyCharm\PycharmProjects\InvoiceRecognition\test_result\result/'


class MyApp(QMainWindow):
    def __init__(self):
        super(MyApp, self).__init__()
        self.fileName = None
        self.filePath = None
        self.message = 'ready'
        self.cut_img_m_name = None

        self.useListWidget = UsedListWidget(self)
        self.funcListWidget = FuncListWidget(self)
        self.stackedWidget = StackedWidget(self)
        # self.cutImg = CutImg(self)
        self.showRes = ShowResWidget(self)
        self.graphicsView = GraphicsView(self)

        self.dock_func = QDockWidget(self)
        self.dock_func.setWidget(self.funcListWidget)
        self.dock_func.setTitleBarWidget(QLabel('图像操作'))
        self.dock_func.setFeatures(QDockWidget.NoDockWidgetFeatures)

        self.dock_used = QDockWidget(self)
        self.dock_used.setWidget(self.useListWidget)
        self.dock_used.setTitleBarWidget(QLabel('已选操作'))
        self.dock_used.setFeatures(QDockWidget.NoDockWidgetFeatures)

        self.dock_attr = QDockWidget(self)
        self.dock_attr.setWidget(self.stackedWidget)
        self.dock_attr.setTitleBarWidget(QLabel('属性'))
        self.dock_attr.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.dock_attr.close()

        # self.dock_btn = QDockWidget(self)
        # self.dock_btn.setWidget(self.cutImg)
        # self.dock_attr.setTitleBarWidget(QLabel('功能性按钮'))
        # self.dock_btn.setFeatures(QDockWidget.NoDockWidgetFeatures)

        self.dock_res = QDockWidget(self)
        self.dock_res.setWidget(self.showRes)
        self.dock_res.setTitleBarWidget(QLabel('结果显示'))
        self.dock_res.setFeatures(QDockWidget.NoDockWidgetFeatures)

        self.setCentralWidget(self.graphicsView)
        self.addDockWidget(Qt.TopDockWidgetArea, self.dock_func)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_used)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_attr)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_res)
        # self.addDockWidget(Qt.BottomDockWidgetArea,self.dock_btn)

        self.setWindowTitle('发票识别')
        self.setWindowIcon(QIcon('window/icons/icon_normal/main.png'))
        self.src_img = None
        self.cur_img = None

        self.initMenu()
        self.initTooBar()
        self.status = self.statusBar()
        self.status.showMessage(self.message)
        self.setStatusBar(self.status)

    def initMenu(self):
        # 菜单栏
        openFile = QAction(QIcon('window/icons/icon_normal/打开.png'), '打开文件', self)
        openFile.setShortcut('Ctrl+o')
        openFile.setStatusTip('Open new File')

        saveFile = QAction(QIcon('window/icons/icon_normal/保存.png'), '保存文件', self)
        saveFile.setShortcut('Ctrl+s')
        saveFile.setStatusTip('save File')

        exitAction = QAction(QIcon(''), '退出', self)
        exitAction.setShortcut('Ctrl+q')
        exitAction.setStatusTip('quit')

        rightRotate = QAction(QIcon('window/icons/icon_normal/右旋转.png'), '右旋转', self)
        rightRotate.setStatusTip('将图片向右旋转90度')

        leftRotate = QAction(QIcon('window/icons/icon_normal/左旋转.png'), '左旋转', self)
        leftRotate.setStatusTip('将图片向左旋转90度')

        getCenterPicture = QAction(QIcon('window/icons/icon_worth/'+ str(random.randint(1,16)) + '.png'), '获取中心图片', self)
        getCenterPicture.setStatusTip('得到图片轮廓中间图片')

        resizeImg = QAction(QIcon('window/icons/icon_worth/'+ str(random.randint(1,16)) + '.png'), '重置图片大小', self)
        resizeImg.setStatusTip('得到发票图片')

        openFile.triggered.connect(self.showFile)
        saveFile.triggered.connect(self.saveFile)
        exitAction.triggered.connect(qApp.quit)
        rightRotate.triggered.connect(self.right_rotate)
        leftRotate.triggered.connect(self.left_rotate)
        getCenterPicture.triggered.connect(self.get_center_picture)
        resizeImg.triggered.connect(self.resize_img)

        menuBar = self.menuBar()
        menuBar.setMinimumHeight(30)
        fileMenu = menuBar.addMenu('&文件')
        fileMenu.addAction(openFile)
        fileMenu.addAction(saveFile)
        fileMenu.addAction(exitAction)

        editMenu = menuBar.addMenu('&编辑')

        editMenu.addAction(rightRotate)
        editMenu.addAction(leftRotate)
        editMenu.addAction(resizeImg)
        editMenu.addAction(getCenterPicture)

    def initTooBar(self):
        tool_bar = self.addToolBar('工具栏')
        action_open_file = QAction(QIcon('window/icons/icon_normal/打开.png'), '打开文件', self)
        action_save_file = QAction(QIcon('window/icons/icon_normal/保存.png'), '保存文件', self)
        action_right_rotate = QAction(QIcon("window/icons/icon_normal/右旋转.png"), "向右旋转90", self)
        action_left_rotate = QAction(QIcon("window/icons/icon_normal/左旋转.png"), "向左旋转90°", self)
        action_remove_stamp = QAction(QIcon("window/icons/icon_normal/计算.png"), "去章处理", self)
        # action_get_center_picture =  QAction(QIcon('window/icons/icon_normal/获取图形.png'), '获取中心图片', self)
        # action_resize_picture = QAction(QIcon('window/icons/icon_normal/Resize Image.png'),'图片重置大小',self)
        action_cut_picture_e = QAction(QIcon('window/icons/icon_normal/获取图形.png'), '切割电子发票', self)
        action_cut_picture_m = QAction(QIcon('window/icons/icon_normal/获取图形.png'), '切割发票照片', self)

        # action_cut  = QComboBox(self)
        # action_cut.addAction(action_cut_picture_e)
        # action_cut.addAction(action_cut_picture_m)

        action_recongnize = QAction(QIcon('window/icons/icon_normal/文字识别.png'),'识别发票',self)

        action_show_picture = QAction(QIcon('window/icons/icon_normal/显示.png'),'显示结果',self)

        action_open_file.triggered.connect(self.showFile)
        action_save_file.triggered.connect(self.saveFile)
        action_right_rotate.triggered.connect(self.right_rotate)
        action_left_rotate.triggered.connect(self.left_rotate)
        action_remove_stamp.triggered.connect(self.remove_stamp)
        # action_get_center_picture.triggered.connect(self.get_center_picture)
        # action_resize_picture.triggered.connect(self.resize_img)
        action_cut_picture_e.triggered.connect(self.cut_cur_image_e)
        action_cut_picture_m.triggered.connect(self.cut_cur_image_m)
        action_recongnize.triggered.connect(self.recongnize)
        action_show_picture.triggered.connect(self.getRes)

        tool_bar.addActions((action_open_file,action_save_file,
                             action_left_rotate, action_right_rotate,
                             action_remove_stamp,
                             # action_resize_picture,action_get_center_picture,
                             action_cut_picture_e,action_cut_picture_m,
                             action_recongnize,action_show_picture))
        # tool_bar.addWidget(action_cut)
        tool_bar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)




    def showFile(self):
        file_name= QFileDialog.getOpenFileName(self,'Open file', './test_images','Image files(*.jpg *.gif *.png)')
        self.fileName = os.path.basename(file_name[0]).split('.')[0]
        self.filePath = os.path.dirname(file_name[0])
        if file_name[0].endswith(('.jpg', '.png', '.bmp','jpeg')):
            src_img = cv2.imdecode(np.fromfile(file_name[0], dtype=np.uint8), -1)
            self.change_image(src_img)
        try:
            shutil.rmtree(r'D:\Devsoft\PyCharm\PycharmProjects\InvoiceRecognition\window\img_changed')
            os.mkdir(r'D:\Devsoft\PyCharm\PycharmProjects\InvoiceRecognition\window\img_changed')
        except:
            pass
        self.status.showMessage('成功打开文件')
        self.setStatusBar(self.status)

    def saveFile(self):
        self.graphicsView.save_current()
        self.status.showMessage('成功保存文件')
        self.setStatusBar(self.status)

    def update_image(self):
        if self.src_img is None:
            return
        img = self.process_image()
        self.cur_img = img
        self.graphicsView.update_image(img)

    def change_image(self, img):
        self.src_img = img
        img = self.process_image()
        self.cur_img = img
        self.graphicsView.change_image(img)

    def process_image(self):
        img = self.src_img.copy()
        for i in range(self.useListWidget.count()):
            img = self.useListWidget.item(i)(img)
        return img

    def right_rotate(self):
        img = self.cur_img
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)

        gW = self.graphicsView.width()
        gH = self.graphicsView.height()

        M = cv2.getRotationMatrix2D(center, -90, 1.0)
        rotated = cv2.warpAffine(img, M, (w,h))
        self.change_image(rotated)

        self.status.showMessage('向右旋转成功')
        self.setStatusBar(self.status)

    def left_rotate(self):
        img = self.cur_img
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)

        gW = self.graphicsView.width()
        gH = self.graphicsView.height()

        M = cv2.getRotationMatrix2D(center, 90, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h))
        self.change_image(rotated)
        self.status.showMessage('向左旋转成功')
        self.setStatusBar(self.status)

    def remove_stamp(self):
        img = self.cur_img
        B_channel, G_channel, R_channel = cv2.split(img)  #BGR
        _, RedThresh = cv2.threshold(R_channel, 170, 355, cv2.THRESH_BINARY)
        self.change_image(RedThresh)

    def cut_cur_image_e(self):
        cut_image_e(self.cur_img)
        self.status.showMessage('切割成功')
        self.setStatusBar(self.status)

    def cut_cur_image_m(self):
        self.cut_img_m_name = cut_img_m(self.cur_img,'desTest')
        self.status.showMessage('切割成功')
        self.setStatusBar(self.status)


    def get_center_picture(self):
        img = self.cur_img
        B_channel, G_channel, R_channel = cv2.split(img)  # BGR
        _, thresh = cv2.threshold(R_channel, 170, 355, cv2.THRESH_BINARY)

        edged = cv2.Canny(thresh, 100, 200)
        contours, hierarchy = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        arc_map = {}
        for cnt in contours:
            arc = cv2.arcLength(cnt,True)
            arc_map[arc] = cnt
        sorted_arc = sorted(arc_map.keys())


        c = arc_map.get(sorted_arc[-1])
        (x,y,w,h) = cv2.boundingRect(c)
        arr = np.array(img)
        center_img = arr[y:y+h,x:x+w]
        self.change_image(center_img)
        cv2.imwrite('window/img_changed/center_picture.png',center_img)

        self.status.showMessage('成功得到中心图片')
        self.setStatusBar(self.status)



    def recongnize(self):
        recognition.recongnize()
        self.status.showMessage('识别成功')
        self.setStatusBar(self.status)

    def resize_img(self):
        img = self.src_img
        img = cv2.resize(img, (0, 0), fx=0.5,fy=0.25,interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        width = self.graphicsView.width()
        height = self.graphicsView.height()
        # width, height = gray.shape

        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        gradx = cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=-1)
        grady = cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=-1)
        gradx = cv2.convertScaleAbs(gradx)
        grady = cv2.convertScaleAbs(grady)
        gradxy = cv2.addWeighted(gradx, 0.5, grady, 0.5, 0)

        gradxy = cv2.morphologyEx(gradxy, cv2.MORPH_CLOSE, rectKernel)

        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 3)
        # _,thresh = cv2.threshold(gradxy,127,255,cv2.THRESH_BINARY)
        # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, rectKernel)
        edged = cv2.Canny(thresh, 100, 200)
        contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(img,contours,-1,(0,255,0),2)

        arc_map = {}
        for cnt in contours:
            arc = cv2.arcLength(cnt, True)

            arc_map[arc] = cnt
        sorted_arc = sorted(arc_map.keys())

        c = arc_map[sorted_arc[-1]]
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.drawContours(gray, c, -1, (0, 255, 0), 2)

        pts1 = np.float32([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
        pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        PerspectiveMatrix = cv2.getPerspectiveTransform(pts1, pts2)
        img = cv2.warpPerspective(img, PerspectiveMatrix, (width, height))

        self.change_image(img)
        self.status.showMessage('重置成功')
        self.setStatusBar(self.status)

    def getRes(self):
        str = res_dir + self.cut_img_m_name + '.txt'
        file = open(str,'r',encoding='utf-8')
        for i in file:
            self.showRes.append(i)
        # self.showRes.toPlainText()
        self.showRes.show()
        self.status.showMessage('结果返回成功')
        self.setStatusBar(self.status)










if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(open('D:\Devsoft\PyCharm\PycharmProjects\CTPN_CRNN_bac\window\custom\styleSheet.qss', encoding='utf-8').read())
    window = MyApp()
    window.show()
    sys.exit(app.exec_())

import cv2
import numpy as np
import math

dir = r'D:\Devsoft\PyCharm\PycharmProjects\InvoiceRecognition\window\img_changed/'

# 水平方向投影
def hProject(binary):
    h, w = binary.shape

    # 水平投影
    # hprojection = np.zeros(binary.shape, dtype=np.uint8)
    hprojection = np.full(binary.shape,255,dtype=np.uint8)

    # 创建h长度都为0的数组
    h_h = [0]*h
    count = 0

    # 统计黑色像素
    # for j in range(0,h,count):
    #     count = 0
    #     for i in range(w):
    #         if binary[j,i] == 0:
    #             h_h[j] += 1
    #     if h_h[j] != 0:
    #         count += 10

    while count < h:
        for i in range(w):
            if binary[count,i] == 0:
                h_h[count] += 1
        if h_h[count] != 0:
            count += 40
        else:
            count += 1
    # 画出投影图
    for j in range(h):
        for i in range(h_h[j]):
            hprojection[j,i] = 0
    # cv2.imshow('hpro', hprojection)
    # cv2.waitKey()

    # h_h 的长度：h h_h 大小 ：w
    return h_h


def get_watershed(img):
    # Step1. 加载图像
    img = cv2.resize(img,(0,0),fx=0.25,fy=0.25,interpolation=cv2.cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Step2.阈值分割，将图像分为黑白两部分
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Step3. 对图像进行“开运算”，先腐蚀再膨胀
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # cv2.imshow("opening", opening)
    # Step4. 对“开运算”的结果进行膨胀，得到大部分都是背景的区域
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # Step5.通过distanceTransform获取前景区域
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)  # DIST_L1 DIST_C只能 对应掩膜为3    DIST_L2 可以为3或者5
    ret, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)
    # Step6. sure_bg与sure_fg相减,得到既有前景又有背景的重合区域   #此区域和轮廓区域的关系未知
    sure_fg = np.uint8(sure_fg)
    unknow = cv2.subtract(sure_bg, sure_fg)
    # Step7. 连通区域处理
    ret, markers = cv2.connectedComponents(sure_fg, connectivity=8)  # 对连通区域进行标号  序号为 0 - N-1
    print(ret)
    markers = markers + 1  # OpenCV 分水岭算法对物体做的标注必须都 大于1 ，背景为标号 为0  因此对所有markers 加1  变成了  1  -  N
    # 去掉属于背景区域的部分（即让其变为0，成为背景）
    # 此语句的Python语法 类似于if ，“unknow==255” 返回的是图像矩阵的真值表。
    markers[unknow == 255] = 0
    # Step8.分水岭算法
    markers = cv2.watershed(img, markers)  # 分水岭算法后，所有轮廓的像素点被标注为  -1
    gray[markers != 1] = 255
    return gray


def DegreeTrans(theta):
    res = theta / np.pi * 180
    return res


# 逆时针旋转图像degree角度（原尺寸）
def rotateImage(src, degree):
    # 旋转中心为图像中心
    h, w = src.shape[:2]
    # 计算二维旋转的仿射变换矩阵
    RotateMatrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), degree, 1)
    print(RotateMatrix)
    # 仿射变换，背景色填充为白色
    rotate = cv2.warpAffine(src, RotateMatrix, (w, h), borderValue=(255, 255, 255))
    return rotate


# 通过霍夫变换计算角度
def CalcDegree(srcImage):
    # midImage = cv2.cvtColor(srcImage, cv2.COLOR_BGR2GRAY)
    dstImage = cv2.Canny(srcImage, 50, 200, 3)
    lineimage = srcImage.copy()
    # 通过霍夫变换检测直线
    # 第4个参数就是阈值，阈值越大，检测精度越高
    lines = cv2.HoughLines(dstImage, 1, np.pi / 180, 200)
    # 由于图像不同，阈值不好设定，因为阈值设定过高导致无法检测直线，阈值过低直线太多，速度很慢
    sum = 0
    # 依次画出每条线段
    for i in range(len(lines)):
        for rho, theta in lines[i]:
            # print("theta:", theta, " rho:", rho)
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(round(x0 + 1000 * (-b)))
            y1 = int(round(y0 + 1000 * a))
            x2 = int(round(x0 - 1000 * (-b)))
            y2 = int(round(y0 - 1000 * a))
            # 只选角度最小的作为旋转角度
            sum += theta
            cv2.line(lineimage, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)
            # cv2.imshow("Imagelines", lineimage)
    # 对所有角度求平均，这样做旋转效果会更好
    average = sum / len(lines)
    angle = DegreeTrans(average) - 90
    return angle

def cut_img_m(img,imgName):
    gray = get_watershed(img)
    try:
        degree = CalcDegree(gray)
        print("调整角度：", degree)
        rotate = rotateImage(gray, degree)
        # cv2.imshow("rotate", rotate)
        # cv2.imwrite("recified.png", rotate, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    except:
        pass
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 3)

    # 不同参数腐蚀
    scale = 22
    h_size = int(len(gray[1]) / scale)
    v_size = int(len(gray[0]) / scale)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_size, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_size))

    h_th = th.copy()
    w_th = th.copy()


    h_th = cv2.morphologyEx(h_th, cv2.MORPH_CLOSE, h_kernel)
    # w_th = cv2.morphologyEx(w_th, cv2.MORPH_CLOSE, v_kernel)


    h_h = hProject(h_th)

    # cv2.imshow('th', h_th)
    # cv2.waitKey()
    # cv2.imshow('w_th', w_th)
    #
    # cv2.imshow('gray',gray)

    h, w = th.shape
    start = True
    h_start, h_end = [], []

    index = 0

    # 根据水平投影获取垂直分割
    for i in range(len(h_h)):

        # 有黑色，即当做头
        if h_h[i] == 0 and start:
            h_start.append(i)
            start = False
            # 全白，即结束
        if h_h[i] > 0 and not start:
            h_end.append(i)
            start = True

    for i in range(len(h_start)):
        if i == len(h_start) - 1:
            cropImg = gray[h_start[i]:, 0:w]
        else:
            cropImg = gray[h_start[i]:h_end[i], 0:w]

        if cropImg.shape[0] > 20:
            cv2.imwrite(dir + imgName + '_crop_' + str(i) + '.png', cropImg)

        # if i == 4 or i == 7:
        #     index = cut_side(index,cropImg,v_kernel)
        # elif i == 5:
        #     index = cut_center(index,cropImg,v_kernel)
    return imgName


if __name__ == '__main__':
    img = cv2.imread('./test_images/03.jpg')
    cut_img_m(img,'test')
    # get_watershed(img)

import cv2
import numpy as np

dir = r'D:\Devsoft\PyCharm\PycharmProjects\InvoiceRecognition\window\img_changed/'

# 水平方向投影
def hProject(binary):
    h, w = binary.shape

    # 水平投影
    hprojection = np.zeros(binary.shape, dtype=np.uint8)

    # 创建h长度都为0的数组
    h_h = [0]*h
    # 统计黑色像素
    for j in range(h):
        for i in range(w):
            if binary[j,i] == 0:
                h_h[j] += 1
    # 画出投影图
    for j in range(h):
        for i in range(h_h[j]):
            hprojection[j,i] = 0

    # cv2.imshow('hpro', hprojection)
    # cv2.waitKey(0)

    # h_h 的长度：h h_h 大小 ：w
    return h_h

# 垂直反向投影
def vProject(binary):
    h, w = binary.shape
    # 垂直投影
    vprojection = np.zeros(binary.shape, dtype=np.uint8)

    # 创建 w 长度都为0的数组
    w_w = [0]*w
    for i in range(w):
        for j in range(h):
            if binary[j, i ] == 0:
                w_w[i] += 1

    for i in range(w):
        for j in range(w_w[i]):
            vprojection[j,i] = 255

    # cv2.imshow('vpro', vprojection)
    # cv2.waitKey(0)

    return w_w


def cut_center(index,thresh):
    '''
    截取货物名称栏
    :param index:  索引，需要对图片进行索引
    :param thresh:  传入需要处理的二值图片
    :param w_w:  垂直投影
    :return:  返回索引
    '''
    start = True
    v_start, v_end = [], []
    h,w = thresh.shape
    w_w = vProject(thresh)
    for i in range(len(w_w)):
        if w_w[i] >= h / 2:

            if start:
                v_start.append(i)
                start = False
                continue
            if not start:
                v_end.append(i)
                v_start.append(i)

    # # 最后一个不要
    # for i in range(len(v_start) - 1):
    #     cropImg = thresh[:, v_start[i]:v_end[i]]
    #     cv2.imwrite(dir +'subImg' + str(index) + '.jpg', cropImg)
    #     index += 1

    # 更改版本：值获取第一个，物品名称
    cropImg = thresh[:, v_start[0]:v_end[0]]
    cv2.imwrite(dir +'e_centerImg' + str(index) + '.jpg', cropImg)
    index += 1
    return index

def cut_side(index,thresh):
    start = True
    v_start,v_end = [],[]
    h,w = thresh.shape
    w_w = vProject(thresh)
    isSingel = True
    for i in range(len(w_w)):
        if w_w[i] >= 2 * h / 3:
            if isSingel:
                isSingel = False
                continue

            if start:
                v_start.append(i)
                start = False
                continue
            if not start:
                v_end.append(i)
                start = True


    # for i in range(len(v_start)):
    #     cropImg = thresh[:,v_start[i]:v_end[i]]
    #     cv2.imwrite(dir + 'subImg'+ str(index) + '.png', cropImg)
    #     index += 1
    cropImg = thresh[:, v_start[0]:v_end[0]]
    cv2.imwrite(dir + 'e_sideImg'+ str(index) + '.jpg', cropImg)
    index += 1
    return index

def cut_head(index,thresh):
    start = True
    v_start,v_end = [],[]
    h,w = thresh.shape

    # 膨胀腐蚀
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 40))
    new_thresh = thresh.copy()
    new_thresh = cv2.morphologyEx(new_thresh, cv2.MORPH_OPEN, v_kernel)

    w_w = vProject(new_thresh)

    for i in range(len(w_w)):

            if  w_w[i] >= h/2 and start :
                v_start.append(i)
                start = False
            if  w_w[i] == 0 and not start :
                v_end.append(i)
                start = True
    # 不要中间的
    for i in range(len(v_start)):
        if i == 0 or i == len(v_start) - 1:
            cropImg = thresh[:,v_start[i]:v_end[i]]
            cv2.imwrite(dir + 'subImg'+ str(index) + '.png', cropImg)
            index += 1
    return index

def cut_image_e(img):
    # img = cv2.imread(imgPath)

    B_channel, G_channel, R_channel = cv2.split(img)  # BGR
    _, th = cv2.threshold(R_channel, 204, 255, cv2.THRESH_BINARY)

    index = 0

    h, w = th.shape
    h_h = hProject(th)
    start = True
    h_start, h_end = [], []

    # 根据水平投影获取垂直分割
    for i in range(len(h_h)):

        # 有黑色，即当做头
        if h_h[i] > 0 and start:
            h_start.append(i)
            start = False
            # 全白，即结束
        if h_h[i] >= w / 2 and not start:
            h_end.append(i)
            start = True

    for i in range(len(h_start)):
        if i == len(h_start) - 1:
            cropImg = th[h_start[i]:, 0:w]
        else:
            cropImg = th[h_start[i]:h_end[i], 0:w]

        cv2.imwrite('cropImg' + str(i) + '.png', cropImg)


        # if i == 0:
        #     index = cut_head(index, cropImg)
        if i == 2:
            index = cut_center(index, cropImg)
        elif i == 1 or i == 4:
            index = cut_side(index,cropImg)
        # elif i == 5:
        #     continue
        # else:
        #     index = cut_side(index, cropImg)

if __name__ == '__main__':
    img = cv2.imread('./test_images/02.png')
    cut_image_e(img)
#     img = cv2.imread('./test_images/02.png')
#
#     B_channel, G_channel, R_channel = cv2.split(img)  # BGR
#     _, th = cv2.threshold(R_channel, 204, 255, cv2.THRESH_BINARY)
#
#     index = 0
#
#     # h,w = th.shape
#     #
#     # w_w = vProject(th)
#     #
#     # cut_center(index,th,w_w,h)
#
#
#
#     h,w = th.shape
#     h_h = hProject(th)
#     start = 0
#     h_start, h_end = [], []
#
#     # 根据水平投影获取垂直分割
#     for i in range(len(h_h)):
#
#         # 有黑色，即当做头
#         if h_h[i] > 0 and start == 0:
#             h_start.append(i)
#             start = 1
#             # 全白，即结束
#         if h_h[i] >= w / 2 and start == 1:
#             h_end.append(i)
#             start = 0
#
#     for i in range(len(h_start)):
#         if i == len(h_start) - 1:
#             cropImg = th[h_start[i]:, 0:w]
#         else:
#             cropImg = th[h_start[i]:h_end[i], 0:w]
#
#         cv2.imwrite('cropImg' + str(i) + '.png',cropImg)
#
#         w_w = vProject(cropImg)
#
#         if i == 0:
#             index = cut_head(index,cropImg)
#         elif i == 2:
#             index = cut_center(index,cropImg)
#         elif i == 5:
#             continue
#         else:
#             index = cut_side(index,cropImg)
#     #
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
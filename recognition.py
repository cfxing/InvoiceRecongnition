import os
import time
import numpy as np
from glob import glob
from PIL import Image
from ocr import ocr
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def single_pic_proc(image_file):
    image = np.array(Image.open(image_file).convert('RGB'))
    h = image.shape[0]
    result, image_framed = ocr(image,h)
    return result, image_framed


def recongnize():
    image_files = glob(r'D:\Devsoft\PyCharm\PycharmProjects\InvoiceRecognition\window\img_changed\*.*')
    result_dir = './test_result/result/'
    for image_file in sorted(image_files):
        t = time.time()
        result, image_framed = single_pic_proc(image_file)
        output_file = os.path.join(result_dir, image_file.split('\\')[-1])
        txt_file = os.path.join(result_dir, image_file.split('\\')[-1].split('.')[0].split('_')[0] + '.txt')

        print(txt_file)
        txt_f = open(txt_file, 'a',encoding='utf-8')
        Image.fromarray(image_framed).save(output_file)
        print("Mission complete, it took {:.3f}s".format(time.time() - t))
        print("\nRecognition Result:\n")
        for key in result:
            print(result[key][1])
            txt_f.write(result[key][1] + '\t')
        txt_f.write('\n')
        txt_f.close()



# if __name__ == '__main__':
#     recongnize()
#
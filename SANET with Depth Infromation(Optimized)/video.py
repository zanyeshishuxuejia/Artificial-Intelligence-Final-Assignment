#%%
from Eval import *
import cv2
import numpy as np
from torchvision.utils import save_image
from PIL import Image
clicked = False


def on_mouse(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONUP:
        clicked = True
if __name__ == '__main__':
    style_path='F:/学习所用/2021年上半年/深度学习/期末大作业/SANET-master/style/la_muse.jpg'
    #vc = cv2.VideoCapture('1 斧头帮跳舞.mp4')  # 读入视频文件，命名cv
    vc = cv2.VideoCapture(0)
    n = 1  # 计数


    if vc.isOpened():  # 判断是否正常打开
        rval, frame = vc.read()
    else:
        rval = False
    fps = 25
    timeF = 1  # 视频帧计数间隔频率
    videoWrite = cv2.VideoWriter("output.avi",  cv2.VideoWriter_fourcc(*'DIVX'), 15, (512, 512))
    i = 0
    cv2.namedWindow('MyWindow')
    cv2.setMouseCallback('MyWindow', on_mouse)

    while rval and cv2.waitKey(5) == -1 and not clicked:  # 循环读取视频帧
        rval, frame = vc.read()
        if (n % timeF == 0):  # 每隔timeF帧进行存储操作
            i += 1
            frame1 = cv2.resize(frame, (512, 512),interpolation=cv2.INTER_NEAREST)
            image = Image.fromarray(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
            convert_content=eval(image,style_path)



            save_image(convert_content, 'output.jpg')

            cv2.imshow('MyWindow',cv2.imread('output.jpg'))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            #videoWrite.write(cv2.imread('output.jpg'))


        n = n + 1
    cv2.destroyWindow('MyWindow')
    vc.release()
    videoWrite.release()

# %%

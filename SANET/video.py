#%%
from Eval import *
import cv2
import numpy as np
from torchvision.utils import save_image
from PIL import Image
clicked = False

def transfrom_video(video_path,style_path):
    vc=cv2.VideoCapture(video_path)
    if vc.isOpened():  # 判断是否正常打开
        rval, frame = vc.read()
    else:
        rval = False
    fps = 25
    n=1
    timeF = 1  # 视频帧计数间隔频率
    videoWrite = cv2.VideoWriter("tranfrom_video.avi", cv2.VideoWriter_fourcc(*'DIVX'), 15, (512, 512))
    i = 0
    while rval:  # 循环读取视频帧
        rval, frame = vc.read()
        if (n % timeF == 0):  # 每隔timeF帧进行存储操作
            i += 1
            frame1 = cv2.resize(frame, (512, 512),interpolation=cv2.INTER_NEAREST)
            convert_content=eval(frame1,style_path)
            save_image(convert_content, 'output.jpg')
            videoWrite.write(cv2.imread('output.jpg'))
        n = n + 1
    vc.release()
    videoWrite.release()

def on_mouse(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONUP:
        clicked = True
if __name__ == '__main__':
    style_path='C:/Users/一脚一个小朋友/Desktop/SANET-master/SANET-master/style/Starry.jpg'

    #若要对视频进行风格转换，修改视频路径并运行下列代码
    # transfrom_video('视频.mp4',style_path)

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
            convert_content=eval(frame1,style_path)
            save_image(convert_content, 'output.jpg')
            cv2.imshow('MyWindow',cv2.imread('output.jpg'))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            videoWrite.write(cv2.imread('output.jpg'))
        n = n + 1
    cv2.destroyWindow('MyWindow')
    vc.release()
    videoWrite.release()


# %%

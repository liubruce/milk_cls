import numpy as np
import cv2
from matplotlib import pyplot as plt
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import os
import os.path as osp

def draw_circles(img,circles, title):
    color = (255, 0, 0)
    # Line thickness of -1 px
    thickness = 10
    if circles is not None:
        circles = np.uint16(np.around(circles))
        num = 0
        for i in circles[0, :]:
            # print('i is ', i)
            num = num + 1
            cv2.circle(img, (i[0], i[1]), i[2], color, num*10)
    plt.title(title)
    plt.imshow(img)
    plt.show()

def detect_outside_circle(file_name, if_inner=False):
    img = cv2.imread(file_name)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 25)  # cv2.bilateralFilter(gray,10,50,50)

    minDist = 5000
    param1 = 30  # 500
    param2 = 50  # 200 #smaller value-> more false circles
    minRadius = 0
    maxRadius = 0  # 10

    # docstring of HoughCircles: HoughCircles(image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]]) -> circles
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2,
                               minRadius=minRadius, maxRadius=maxRadius)
    #     circles = cv2.HoughCircles(blurred,cv2.HOUGH_GRADIENT,1,60,param1=50,param2=30,minRadius=0,maxRadius=0)
    #     circles = cv2.HoughCircles(blurred,cv2.HOUGH_GRADIENT,1,minDist,param1=50,param2=30,minRadius=0,maxRadius=0)

    #     print('circles are ', circles)
    # Red color in BGR
    #     color = (0, 0, 255)
    len_circle = 0
    if circles is not None:
        len_circle = len(circles[0])
        print('The number of circle is ', len_circle)
        # if len_circle > 1:
        head, tail = os.path.split(file_name)
        draw_circles(img, circles, tail)
    else:
        _, tail = os.path.split(file_name)
        plt.title('Empty ' + tail)
        plt.imshow(img)
        plt.show()
        print(file_name)
    return file_name, circles, len_circle


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    # print(Path(__file__).resolve())

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # print(str(project_dir) + '/raw/dn', Path(__file__).resolve().parents)
    p_dn = Path(str(project_dir) + '/data/raw/dn').resolve()
    p_pc = Path(str(project_dir) + '/data/raw/pc').resolve()
    for x in p_dn.iterdir():
        # print(x)
        detect_outside_circle(str(x))
    for x in p_pc.iterdir():
        # print(x)
        detect_outside_circle(str(x))
    # get_all_datasets(Path(str(project_dir) + '/raw/dn').iterdir())


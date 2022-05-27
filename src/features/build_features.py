import numpy as np
import cv2
from matplotlib import pyplot as plt
import logging
from pathlib import Path
import os
import skimage.io
import skimage.color
import skimage.filters
from PIL import Image, ImageDraw

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


def detect_circles(pil_image, title):
    open_cv_image = np.array(pil_image.convert('RGB'))
    # Convert RGB to BGR
    img = open_cv_image[:, :, ::-1].copy()
    #     img = cv2.cv.CreateImageHeader(pimg.size,cv2.IPL_DEPTH_8U,3)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     gray = np.asarray(img)
    # plt.imshow(gray)
    # plt.show()
    blurred = cv2.medianBlur(gray, 25)  # cv2.bilateralFilter(gray,10,50,50)

    minDist = 500
    param1 = 30  # 500
    param2 = 50  # 200 #smaller value-> more false circles
    minRadius = 0
    maxRadius = 0  # 10

    # docstring of HoughCircles: HoughCircles(image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]]) -> circles
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2,
                               minRadius=minRadius, maxRadius=maxRadius)
    #     circles = cv2.HoughCircles(blurred,cv2.HOUGH_GRADIENT,1,60,param1=50,param2=30,minRadius=0,maxRadius=0)
    #     circles = cv2.HoughCircles(blurred,cv2.HOUGH_GRADIENT,1,minDist,param1=50,param2=30,minRadius=0,maxRadius=0)

    # print('circles are ', circles)
    # Red color in BGR
    #     color = (0, 0, 255)
    color = (255, 0, 0)
    # Line thickness of -1 px
    thickness = 10

    # Using cv2.circle() method
    # Draw a circle of red color of thickness -1 px
    #     image = cv2.circle(image, center_coordinates, radius, color, thickness)

    if circles is not None:
        len_circle = len(circles[0])
        print('The number of circle is ', len_circle)
        # if len_circle > 1:
        # head, tail = os.path.split(file_name)
        draw_circles(img, circles, title)

        # circles = np.uint16(np.around(circles))
        # num = 1
        # for i in circles[0, :]:
        #     print('i is ', i)
        #     num = num + 1
        #     cv2.circle(img, (i[0], i[1]), i[2], color, num*10)
        #     cv2.circle(img, (i[0], i[1]), 20, color, 3)
    #             if if_inner:
    #                 cv2.circle(img, (i[0], i[1]), i[2]-150, color, num * thickness)

    # Show result for testing:
    # cv2.imshow('img', img)
    #     plt.title(title)
    #     plt.imshow(img)
    #     plt.show()
    else:
        print('Empty : ', title)
        draw_circles(pil_image, circles, 'Empty : '+ title)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def detect_outside_circle(file_name, blur_ksize=25, draw_circle=True):
    img = cv2.imread(file_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, blur_ksize)  # cv2.bilateralFilter(gray,10,50,50)

    minDist = 5000
    param1 = 30  # 500
    param2 = 50  # 200 #smaller value-> more false circles
    minRadius = 0
    maxRadius = 0  # 10
    circle_none_files = []
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
        if draw_circle:
            draw_circles(img, circles, tail)
    else:
        _, tail = os.path.split(file_name)
        return detect_outside_circle(file_name, 5, draw_circle)
        # plt.title('Empty ' + tail)
        # plt.imshow(img)
        # plt.show()
        # print(file_name)
    return circles, len_circle, gray

def detect_inner_ring(file_name):
    # load the image
    image = skimage.io.imread(file_name)
    # image = skimage.io.imread('../data/raw/dn/220207a_C1.bmp')
    # fig, ax = plt.subplots()
    # plt.imshow(image)
    # plt.show()
    # convert the image to grayscale
    gray_image = skimage.color.rgb2gray(image)

    # blur the image to denoise
    blurred_image = skimage.filters.gaussian(gray_image, sigma=1.0)

    # fig, ax = plt.subplots()
    # plt.imshow(blurred_image, cmap='gray')
    # plt.show()

    t = skimage.filters.threshold_otsu(blurred_image)
    # print('Found automatic threshold t = {}.'.format(t))
    # create a mask based on the threshold
    # before 0.3
    # t = 0.3
    binary_mask = blurred_image > t

    # fig, ax = plt.subplots()
    # plt.imshow(binary_mask, cmap='gray')
    # plt.show()

    pixels = np.where(binary_mask, gray_image, blurred_image)

    # Save resulting image
    result = Image.fromarray(pixels)
    # result.save('result.png')
    # plt.imshow(result)
    # plt.show()
    _, tail = os.path.split(file_name)
    # return detect_outside_circle(file_name, 5)
    detect_circles(result, tail)


def getLBPimage(image, BGR2GRAY=True):
    '''
    == Input ==
    gray_image  : color image of shape (height, width)

    == Output ==
    imgLBP : LBP converted image of the same shape as
    '''
    ### Step 0: Step 0: Convert an image to grayscale
    gray_image = image
    if BGR2GRAY:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    imgLBP = np.zeros_like(gray_image)
    neighboor = 3
    for ih in range(0, image.shape[0] - neighboor):
        for iw in range(0, image.shape[1] - neighboor):
            ### Step 1: 3 by 3 pixel
            img = gray_image[ih:ih + neighboor, iw:iw + neighboor]
            center = img[1, 1]
            img01 = (img >= center) * 1.0
            img01_vector = img01.T.flatten()
            # it is ok to order counterclock manner
            # img01_vector = img01.flatten()
            ### Step 2: **Binary operation**:
            img01_vector = np.delete(img01_vector, 4)
            ### Step 3: Decimal: Convert the binary operated values to a digit.
            where_img01_vector = np.where(img01_vector)[0]
            if len(where_img01_vector) >= 1:
                num = np.sum(2 ** where_img01_vector)
            else:
                num = 0
            imgLBP[ih + 1, iw + 1] = num
    return imgLBP

def extract_infor_circle(img_origin, x, y, r):
    img_arr = img_origin
    h,w = img_arr.shape[1], img_arr.shape[0]
    lum_img = Image.new('L',[h,w] ,0)
    draw = ImageDraw.Draw(lum_img)

    leftUpPoint = (x-r, y-r)
    rightDownPoint = (x+r, y+r)
    twoPointList = [leftUpPoint, rightDownPoint]
    draw.ellipse(twoPointList, fill=255)
    lum_img_arr = np.array(lum_img)
    final_img_arr = np.dstack((img_arr, lum_img_arr))
    return final_img_arr

def circle_to_lbp(image_file, draw_circle):
    circles, num_cirlces, img = detect_outside_circle(str(image_file), draw_circle=draw_circle)
    merged_image = extract_infor_circle(img, circles[0][0][0], circles[0][0][1], circles[0][0][2])
    img = getLBPimage(merged_image, False)
    return img

def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles

def create_training_data(image_path, if_circle=False, draw_circle=False):
    imagePaths = getListOfFiles(image_path) ## Folder structure: datasets --> sub-folders with labels name

    data = []
    lables = []
    c = 0 ## to see the progress
    for image in imagePaths:

        lable = os.path.split(os.path.split(image)[0])[1]
        lables.append(lable)
        print(image)
        img = cv2.imread(image)
        if if_circle:
            imge = circle_to_lbp(image, draw_circle)
        else:
            img = getLBPimage(img)
        img = cv2.resize(img, (100, 100), interpolation = cv2.INTER_AREA)
        data.append(img[0])
        c=c+1
    return data, lables

def extract_circle_v2(image, height, width, first_ring, second_ring, only_one=True):
    centerX, centerY, radius = first_ring
    centerX2, centerY2, radius2 = second_ring
    canvas = np.zeros((height, width))
    # Draw the outer circle:
#     print('redius is ', radius, radius2)
    color = (255, 255, 255)
    thickness = -1
#     centerX = i[0]
#     centerY = i[1]
#     radius = i[2]
    if only_one:
        radius = radius2
        cv2.circle(canvas, (centerX, centerY), radius, color, thickness)
    else:
        cv2.circle(canvas, (centerX2, centerY2), radius2, color, thickness)
        color = (0, 0, 0)
        cv2.circle(canvas, (centerX, centerY), radius, color, thickness)

    # Create a copy of the input and mask input:
    imageCopy = image.copy()
    imageCopy[canvas == 0] = (0, 0, 0)

    # Crop the roi:
    x = centerX - radius
    y = centerY - radius
    h = 2 * radius
    w = 2 * radius
    if not only_one:
        x = centerX2 - radius2
        y = centerY2 - radius2
        h = 2 * radius2
        w = 2 * radius2
    croppedImg = imageCopy[y:y + h, x:x + w]
    return croppedImg


def create_lbp_by_rings(data_dir):
    #     i = 0
    vector_data = [[], [], [], [], []]
    labels = []
    num_images = 0
    for image_file in data_dir.iterdir():
        lable = os.path.split(os.path.split(image_file)[0])[1]
        labels.append(lable)
        #         if i > 1:
        #             break;
        #         print(os.path.basename(image_file))
        #         print(image_file)
        image = cv2.imread(str(image_file))
        #         print(image.shape)
        x, y, r = round(image.shape[0] / 2), round(image.shape[1] / 2), 190
        height = image.shape[0]
        width = image.shape[1]
        if num_images == 0:
            fig = plt.figure(figsize=(50, 10))
            ax = fig.add_subplot(1, 16, 1)
            ax.set_title(os.path.basename(image_file))
            ax.imshow(image)
            fig_num = 1
        all_circle_data = []
        for j in range(5):
            merged_image = extract_circle_v2(image, height, width, (y, x, j * r), (y, x, (j + 1) * r),
                                             False if j > 0 else True)
            imgLBP = getLBPimage(merged_image, True)
            #             if j == 1:
            vector_data[j].append(imgLBP)
            vecimgLBP = imgLBP.flatten()
            # print('imgLBP.shape', imgLBP.shape)
            #             ax.set_title("gray scale image")
            if num_images == 0:
                fig_num += 1
                ax = fig.add_subplot(1, 16, fig_num)
                ax.imshow(Image.fromarray(merged_image))
                ax.set_title("gray scale image")
                fig_num += 1
                ax = fig.add_subplot(1, 16, fig_num)
                ax.imshow(imgLBP, cmap='gray', vmin=0, vmax=255)
                ax.set_title("LBP image")
                fig_num += 1
                ax = fig.add_subplot(1, 16, fig_num)
                freq, lbp, _ = ax.hist(vecimgLBP, bins=2 ** 8)
                ax.set_ylim(0, 100000)
                lbp = lbp[:-1]
                ax.set_title("LBP histogram" + str((j + 1) * r))
        if num_images == 0:
            plt.show()
        num_images += 1

    return vector_data, labels

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    # print(Path(__file__).resolve())

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # print(str(project_dir) + '/raw/dn', Path(__file__).resolve().parents)
    # empty_files = []
    # empty_files.append('/Users/bruceliu/projects/milk_cls/data/raw/pc/LBM_70_5Jan_PC.jpg')
    # empty_files.append('/Users/bruceliu/projects/milk_cls/data/raw/pc/LBM_70B_5Jan_PC.jpg')
    # empty_files.append('/Users/bruceliu/projects/milk_cls/data/raw/pc/220208a_DBM_5uL_38C_C1.bmp')
    # empty_files.append('/Users/bruceliu/projects/milk_cls/data/raw/pc/220208a_DBM_5uL_38C_C1_Photo 2.bmp')
    # empty_files.append('/Users/bruceliu/projects/milk_cls/data/raw/pc/DBM_58_13Jan_PC.jpg')
    # for file in empty_files:
    #     detect_outside_circle(file, 5)
    # exit(0)


    p_dn = Path(str(project_dir) + '/data/raw/dn').resolve()
    p_pc = Path(str(project_dir) + '/data/raw/pc').resolve()



    # exit(0)

    circle_info = []
    for x in p_dn.iterdir():
        # print(x)
        circles, num_cirlces = detect_outside_circle(str(x))
        circle_info.append([circles, num_cirlces, str(x)])
        detect_inner_ring(str(x))
    # exit(0)
    for x in p_pc.iterdir():
        # print(x)
        circles, num_cirlces = detect_outside_circle(str(x))
        circle_info.append([circles, num_cirlces, str(x)])
        detect_inner_ring(str(x))

    print(circle_info)

    # circle_info = []
    for x in p_dn.iterdir():
        # print(x)
        # circles, num_cirlces = detect_outside_circle(str(x))
        # circle_info.append([circles, num_cirlces, str(x)])
        detect_inner_ring(str(x))
    # exit(0)
    for x in p_pc.iterdir():
        # print(x)
        # circles, num_cirlces = detect_outside_circle(str(x))
        # circle_info.append([circles, num_cirlces, str(x)])
        detect_inner_ring(str(x))

    # print(circle_info)
    # get_all_datasets(Path(str(project_dir) + '/raw/dn').iterdir())


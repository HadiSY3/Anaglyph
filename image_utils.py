import numpy as np
import cv2 as cv
from screeninfo import get_monitors


    
def resize_to_width(img):
    target_width = 1440
    height, width = img.shape[:2]
    scale_factor = target_width / width
    new_height = int(height * scale_factor)
    return cv.resize(img, (target_width, new_height), interpolation=cv.INTER_AREA)

def shiftImage(img, max_rand, min_rand):
    h, w = img.shape[:2]
    
    for m in get_monitors():
        ppmm = (m.width)/(m.width_mm) 
        if m.height/h > m.width/w:
            scale_factor = m.width/w
        else:
            scale_factor = m.height/h
            
    shift = (scale_factor*min_rand/ppmm) * ppmm
    '''
    print("Maximaler Versatz in Pixel: ", scale_factor*max_rand)
    print("Maximaler Versatz in mm: ", scale_factor*max_rand/ppmm)
    print("Minimaler Versatz in Pixel: ", scale_factor*min_rand)
    print("Minimaler Versatz in mm: ", scale_factor*min_rand/ppmm)
    print("shift in pixel: ", shift)
    print("shift in mm: ", shift/ppmm)
    '''

    M = np.float32([[1, 0, shift], [0, 1, 0]])  
    return cv.warpAffine(img, M, (w,h))

def crop(img1_rectified, img2_rectified, H1, H2):
    h, w = img1_rectified.shape[:2]
    corners = np.float32([[0, 0], [w, 0], [0, h], [w, h]]).reshape((4, 1, 2))

    corners1 = np.squeeze(cv.perspectiveTransform(corners, H1))
    corners2 = np.squeeze(cv.perspectiveTransform(corners, H2))

    for corner in corners1:
        if corner[0] < 0: corner[0] = 0
        if corner[0] > w: corner[0] = w

        if corner[1] < 0: corner[1] = 0
        if corner[1] > h: corner[1] = h


    for corner in corners2:
        if corner[0] < 0: corner[0] = 0
        if corner[0] > w: corner[0] = w

        if corner[1] < 0: corner[1] = 0
        if corner[1] > h: corner[1] = h

    y_oben  = np.max([corners1[0][1], corners2[0][1], corners1[1][1], corners2[1][1]])
    y_unten = np.min([corners1[2][1], corners2[2][1], corners1[3][1], corners2[3][1]])
    x_links  = np.max([corners1[0][0], corners2[0][0], corners1[2][0], corners2[2][0]])
    x_rechts = np.min([corners1[1][0], corners2[1][0], corners1[3][0], corners2[3][0]])

    rect = np.array([[x_links, y_oben],
                        [x_links, y_unten], 
                        [x_rechts, y_oben],
                        [x_rechts, y_unten]])


    cropped1 = img1_rectified[int(y_oben):int(y_unten), int(x_links):int(x_rechts), :]
    cropped2 = img2_rectified[int(y_oben):int(y_unten), int(x_links):int(x_rechts), :]
    return cropped1, cropped2

def load_images(left_path, right_path):
    img1 = cv.imread(left_path)
    img2 = cv.imread(right_path)

    img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
    img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
    
    img1 = resize_to_width(img1)
    img2 = resize_to_width(img2)
    return img1, img2

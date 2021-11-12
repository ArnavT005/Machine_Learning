import cv2
import numpy as np
from PIL import Image

filename = "train_data/res10.jpg"

def binaryThreshold():
    img = cv2.imread(filename, 0)
    img = cv2.fastNlMeansDenoising(img, None, 10, 7, 15)

    cv2.imshow("Processed Image", img)
    cv2.waitKey(2000)

    # imgf = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # ret, imgf = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY, cv2.THRESH_OTSU)
    ret, imgf = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    imgf = cv2.fastNlMeansDenoising(imgf, None, 10, 7, 15)

    # imgf = cv2.resize(imgf, (256, 256))
    cv2.imshow("Processed Image", imgf)
    cv2.waitKey(2000)

def usePIL():
    img = Image.open(filename)
    data = np.array(img)
    converted = np.where(data >= 253, 0, 255)
    img = Image.fromarray(converted.astype('uint8'))
    img.show()

def useColouredPIL():
    im = Image.open(filename)
    im = im.convert('RGBA')
    data = np.array(im)   # "data" is a height x width x 4 numpy array
    red, green, blue, alpha = data.T # Temporarily unpack the bands for readability
    # Replace white with red... (leaves alpha values alone...)
    white_areas = (red <= 252) & (blue <= 252) & (green <= 252)
    data[..., :-1][white_areas.T] = (0, 0, 0) # Transpose back needed
    im2 = Image.fromarray(data)
    im2.show()

def mixbinaryPIL():
    im = Image.open(filename)
    im = im.convert('RGBA')
    data = np.array(im)   # "data" is a height x width x 4 numpy array
    red, green, blue, alpha = data.T # Temporarily unpack the bands for readability
    # Replace non-white with black... (leaves alpha values alone...)
    white_areas = (red <= 250) & (blue <= 250) & (green <= 250)
    data[..., :-1][white_areas.T] = (0, 0, 0) # Transpose back needed
    im2 = Image.fromarray(data)
    # im2 = im2.filter(ImageFilter.GaussianBlur(radius = 1))
    img = cv2.cvtColor(np.array(im2), cv2.COLOR_RGBA2GRAY)
    # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY, cv2.THRESH_OTSU)

    cv2.imshow("Otsu Threshold Image", img)
    cv2.waitKey(5000)

    kernel = np.ones((4, 4), np.uint8)
    img_erosion = cv2.erode(img, kernel, iterations=1)
    kernel = np.ones((2, 2), np.uint8)
    img_dilation = cv2.dilate(img, kernel, iterations=1)

    cv2.imshow("Eroded Image", img_erosion)
    cv2.waitKey(4000)
    cv2.imshow("Dilated Image", img_dilation)
    cv2.waitKey(4000)

img = cv2.imread(filename, 1)
cv2.imshow("Original Image", img)
cv2.waitKey(3000)

mixbinaryPIL()
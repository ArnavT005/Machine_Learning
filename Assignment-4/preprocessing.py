from skimage import io, transform, color
import numpy as np
import shutil
import time
import os

def preprocess(img_folder, threshold=(220/255)):
    # store current directory
    pwd = os.getcwd()
    # store path of img_folder
    directory = pwd + '/' + img_folder + '/'
    # get all image paths in directory
    img_paths = os.listdir(directory)
    # make directory for processed images
    if os.path.exists(pwd + '/processed_data/'):
        shutil.rmtree(pwd + '/processed_data/')
    os.mkdir('processed_data')
    # statistics
    start = time.time()
    img_counter = 0
    for img_path in img_paths:
        # read image
        img = io.imread(directory + img_path)
        # filter image
        img = color.rgb2gray(img)
        filter = (img >= threshold)
        img = img * filter
        # resize and convert image
        img = transform.resize(img, (256, 256), anti_aliasing=True)
        img = color.gray2rgb(img)
        # save as integer type
        img = img * 255
        img = img.astype(np.uint8)
        # save rotational variants of the image
        io.imsave('processed_data/' + img_path, img, check_contrast=False)
        # statistics
        img_counter += 1
        end = time.time()
        if img_counter % 1000 == 0:
            print("Processed Image Count: " + str(img_counter) + ", Time: " + str(end - start))

preprocess('train_data')
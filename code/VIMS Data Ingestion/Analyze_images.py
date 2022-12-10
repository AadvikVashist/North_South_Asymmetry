import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import pandas as pd
import csv
# function to analyze a single image
def analyze_image(image_path):
    # read in the image
    img = cv2.imread(image_path)
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.blur(gray, (2, 2))
    # detect circles in the image
    circles = cv2.HoughCircles(gray_blurred, 
        method = cv2.HOUGH_GRADIENT_ALT, dp = 1,minDist = 10, param1 = 60,
        param2 = 0.7, minRadius= int(np.sqrt(img.shape[0]*img.shape[1])  / 3), 
        maxRadius = int(np.mean((img.shape[0], img.shape[1]))/2))
    percentage = 1.0
    if circles is not None:
        detected_circles = np.around(circles)
        detected_circles = detected_circles[0]
        pt = detected_circles[np.argmax(detected_circles[:,2])]
        x, y, radius = pt[0], pt[1], pt[2]
        circle_area = pt[2] ** 2 * 3.1415
        xa, xy= np.uint16((x, y))
        xmin = x-radius
        xmax = x+radius
        ymin = y-radius
        ymax = y+radius
        if not any((ymin < 0, xmin < 0, xmax > img.shape[0],ymax > img.shape[1])):
            percentage = 1.0
            theta = np.linspace(0, 2*np.pi, 36)
            xs = [np.around(radius*np.cos(theta) + x)][0]
            ys = [np.around(radius*np.sin(theta) + y)][0]
            points = [i for i in zip(xs, ys)]
        else:
            theta = np.linspace(0, 2*np.pi, int(radius*16))
            xs = [np.around(radius*np.cos(theta) + x)][0]
            ys = [np.around(radius*np.sin(theta) + y)][0]
            points = [i for i in zip(xs, ys)]
            imager = np.zeros(gray.shape, dtype = "uint8")
            cv2.fillPoly(imager, pts =np.int32([points]), color=(255,255,255))
            number_of_white_pix = np.sum(imager == 255)
            a = radius*radius*np.pi
            percentage = number_of_white_pix/a
            if percentage >= 1: percentage = 0.99
        mask = np.zeros(gray.shape, dtype = "uint8")
        cv2.fillPoly(mask, pts =np.int32([points]), color=(255))
        abc= cv2.mean(gray, mask=mask)
        if abc[0] > 60:
            percentage = 0.0
    else:
        percentage = 0.0
        x = -1
        y = -1
        radius = -1
    return img, x, y, radius, percentage
# function to recursively analyze all images in a folder
def get_file_size(filename):
    # get the size of the file in bytes
    file_size = os.path.getsize(filename)
    return file_size
def expected_time_function(start_time, index, length, file_sized, dataset_size):
    percentage_completion = index/length * 0.6 +  file_sized/dataset_size * 0.4
    delta_time = time.time()-start_time
    try:
        expected_time = delta_time / percentage_completion
    except:
        expected_time = 0
    print("image", index, "of", length-1, " | ", np.around(delta_time,1), "/", np.around(expected_time,1), "seconds  | ", np.around(expected_time-delta_time), "seconds left")
def analyze_folder(folder_path):
    results = []
    # iterate through all files in the folder
    a = os.walk(folder_path)
    walked = [os.path.join(dp, f) for dp, dn, fn in os.walk(folder_path) for f in fn if any((f.endswith(".jpg"),f.endswith(".png"), f.endswith(".tif")))]
    length = len(walked)
    start_time = time.time()
    file_sizes = [get_file_size(file) for file in walked]
    curr_size = 0
    total_size = sum(file_sizes)
    for index, filename in enumerate(walked):
        image_path = os.path.join(folder_path, filename)
        image, x,y,radius, match_percentage = analyze_image(image_path)
        results.append([filename, image, x,y,radius, match_percentage])
        curr_size += file_sizes[index]
        expected_time_function(start_time, index, length, curr_size, total_size)
    return results
def show_positive_results(results):
    for i in results:
        filename, image, x,y,radius, match_percentage = i
        if match_percentage == 1:
            cv2.imshow("image",image)
            cv2.waitKey(100)
def write_results(results, csvFile, filepath):
    arr = []
    for i in results:
        filename, image, x,y,radius, match_percentage = i
        cube = filename.replace("\\", "/").split('/')[-1]
        flyby = [f for f in filename.replace("\\", "/").split('/') if "TI" in f][0]
        ' '.join([str(c) for c in csvFile[0]])
        flyby_index = [index for index in range(csvFile.shape[0]) if flyby in ' '.join([str(c) for c in csvFile[index]])][0]
        row = [cube, *csvFile[flyby_index][1::], match_percentage]
        arr.append(row)
    with open(filepath, 'w', newline="") as f:
        # create the csv writer
        writer = csv.writer(f)
        # write a row to the csv file
        for row in arr:
            writer.writerow(row)
    print("rows written")
# specify the parent folder to analyze
parent_folder = "C:/Users/aadvi/Desktop/North_South_Asymmetry/data/Nantes"
# call the function on the parent folder to get the list of results
results = analyze_folder(parent_folder)
# show_positive_results(results)
nantes_csv = np.array(pd.read_csv('C:/Users/aadvi/Desktop/North_South_Asymmetry/data/flyby_info/nantes.csv'), dtype = 'O')
write_results(results, nantes_csv, "C:/Users/aadvi/Desktop/North_South_Asymmetry/data/flyby_info/nantes_cubes.csv")
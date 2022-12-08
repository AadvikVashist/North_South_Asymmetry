import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
# function to analyze a single image
def analyze_image(image_path):
    # read in the image
    img = cv2.imread(image_path)
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.blur(gray, (10, 10))
    # detect circles in the image
    circles = cv2.HoughCircles(gray_blurred, 
                   cv2.HOUGH_GRADIENT, 1, 20, param1 = 50,
               param2 = 30, minRadius= int(np.sqrt(img.shape[0]*img.shape[1])  / 3))
    if circles is not None:
        detected_circles = np.around(circles)
        detected_circles = detected_circles[0]
        pt = detected_circles[np.argmax(detected_circles[:,2])]
        x, y, radius = pt[0], pt[1], pt[2]
        circle_area = pt[2] ** 2 * 3.1415
        xa, xy= np.uint16((x, y))
        cv2.circle(img,(xa,xy), np.uint16(radius), (0, 255, 0), 2)
            # Draw a small circle (of radius 1) to show the center.
        # cv2.circle(img, (a, b), 1, (0, 0, 255), 3)
        xmin = x-radius
        xmax = x+radius
        ymin = y-radius
        ymax = y+radius
        if not any((ymin < 0, xmin < 0, xmax > img.shape[0],ymax > img.shape[1])):
            cv2.imshow("Detected Circle", img)
            cv2.waitKey(0)
        else:
            print("skipped over", image_path)
    
#    findIntersection

# function to recursively analyze all images in a folder
def analyze_folder(folder_path):
    results = []
    # iterate through all files in the folder
    for filename in os.listdir(folder_path):
        # if the file is an image, analyze it and append the result to the results list
        if (filename.endswith(".jpg") or filename.endswith(".png")):
            image_path = os.path.join(folder_path, filename)
            result = analyze_image(image_path)
            results.append(result)
            # if the file is a folder, recursively call the function on that folder
        elif os.path.isdir(os.path.join(folder_path, filename)):
            subfolder_results = analyze_folder(os.path.join(folder_path, filename))
    return results
def findIntersection(radius, center, line_endpoints):
    # calculate the slope and y-intercept of the line
    m = (line_endpoints[1][1] - line_endpoints[0][1]) / (line_endpoints[1][0] - line_endpoints[0][0])
    b = line_endpoints[0][1] - m * line_endpoints[0][0]
    # calculate the coefficients of the quadratic equation
    a = 1 + m**2
    c = center[0]**2 + center[1]**2 - radius**2 + b**2 - 2*center[1]*b
    d = (2*center[0]*b) - (2*center[0]*center[1]*m) - (2*center[1]*b)

    # calculate the intersection points using the quadratic formula
    x1 = (-d + math.sqrt(d**2 - 4*a*c)) / (2*a)
    y1 = m*x1 + b
    x2 = (-d - math.sqrt(d**2 - 4*a*c)) / (2*a)
    y2 = m*x2 + b

    return (x1, y1), (x2, y2)
# specify the parent folder to analyze
parent_folder = "C:/Users/aadvi/Desktop/North_South_Asymmetry/data/Nantes"

# call the function on the parent folder to get the list of results
results = analyze_folder(parent_folder)
print(results)
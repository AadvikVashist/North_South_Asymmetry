import pickle
import requests
from bs4 import BeautifulSoup
import os
import numpy as np
import time
import urllib.request
from html_table_parser.parser import HTMLTableParser
from urllib.parse import urljoin
import cv2
import matplotlib.pyplot as plt
import time
import pandas as pd
import csv
import concurrent.futures

# for converting the parsed data in a
# pandas dataframe
def read_a_url(url):
    req = urllib.request.Request(url=url, headers={'User-Agent': 'Chrome/111.0.5563.111'})
    index = 0
    while True:
        if index > 10:
            return None
        try:
            f = urllib.request.urlopen(req)
            break
        except:
            print("stuck at",url)
            index+=1
    # reading contents of the website
    return f.read()
def get_file_size(filename):
    # get the size of the file in bytes
    file_size = os.path.getsize(filename)
    return file_size
def parse_nantes_for_all_flybys(search_directory):
    reqs = requests.get(searchurl)
    soup = BeautifulSoup(reqs.text, 'html.parser')
    abc = soup.find_all('a')
    allurls = [urljoin(search_directory,link.get('href')) for link in abc]
    # url filtering
    urls = []
    for link in allurls:
        if "/flyby" in link:
            urls.append(link)
    if urls:
        print("found flybys")
    else:
        print("no flybys found")
    return urls
def expected_time_function(start_time, index, length, file_sized, dataset_size):
    percentage_completion = index/length * 0.6 +  file_sized/dataset_size * 0.4
    delta_time = time.time()-start_time
    try:
        expected_time = delta_time / percentage_completion
    except:
        expected_time = 0
    print("image", index, "of", length-1, " | ", np.around(delta_time,1), "/", np.around(expected_time,1), "seconds  | ", np.around(expected_time-delta_time), "seconds left")

def parse_for_the_info_tables(urllist):
    p = HTMLTableParser()
    start = time.time()
    length = len(urllist)
    for index,url in enumerate(urllist):
        html = read_a_url(url).decode('utf-8')
        p.feed(html)
        end = time.time() - start
        print(url, "                                                                                            ")
        print(url, "passed time",str(end) + "/" +str(length/(index+1)*end) + " seconds             ", end = "/r")
    return [p.tables], np.array(p.tables)
def combine_table_with_url(flybyarray, urls) -> dict:
    return {url :flybyarray[index] for index, url in enumerate(urls)} 
def write_flyby_data(flybyarray, csv_save_path):
    indexOfFlybyIndex = 2
    infexOfFlybyDate = 0
    indexOfCubeQuantity = 3
    indexOfAltitude = 4
    indexOfCassiniMission = 5
    indexOfSequence = 6
    indexOfRevolution = 7
    with open(csv_save_path, 'w', newline = "") as csvfile:
        writer = csv.writer(csvfile, delimiter = ",")
        writer.writerow(["flyby_url" ,"flyby_index","flyby_cassini_mission","flyby_date","flyby_altitude","flyby_cube_quantity","flyby_sequence","flyby_revolution"])
        for index, (url,flyby) in enumerate(flybyarray.items()):
            writer.writerow([url, flyby[indexOfFlybyIndex][1], flyby[indexOfCassiniMission][1], flyby[infexOfFlybyDate][1],flyby[indexOfAltitude][1],flyby[indexOfCubeQuantity][1],flyby[indexOfSequence][1],flyby[indexOfRevolution][1]])
def analyze_image_for_visibility(image_path):
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
            canny = cv2.Canny(img, 10,80)
            canny = cv2.blur(canny, (5,5))
            # kernel = np.ones((2,2),np.uint8)
            # canny = cv2.dilate(canny,kernel,iterations = 5)
            # kernel = np.ones((3,3),np.uint8)
            # canny = cv2.dilate(canny,kernel,iterations = 1)
            canny = [[255 if c > 0 else 0 for c in r] for r in canny ]
            canny = np.array(canny, dtype=np.uint8)
            cannys = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
            plt.imshow(cv2.bitwise_and(img,cannys))
            
            contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # plt.pause(0.1)
            # plt.imshow(cannys)
            # plt.pause(0.1)

    else:
        percentage = 0.0
        x = -1
        y = -1
        radius = -1

    return img, x, y, radius, percentage



def parse_url(url):
    html = read_a_url(url).decode('utf-8')
    p = HTMLTableParser()
    p.feed(html)
    return p.tables

def work_parse_for_the_info_tables(urllist):
    p = HTMLTableParser()
    start = time.time()
    length = len(urllist)
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(read_a_url, url) for url in urllist]
        for index, future in enumerate(concurrent.futures.as_completed(futures)):
            if future.result() is not None:
                html = future.result().decode('utf-8')
                p.feed(html)
                end = time.time() - start
                print(urllist[index], "                                                                                            ")
                print(urllist[index], "passed time", str(end) + "/" + str(length / (index + 1) * end) + " seconds             ", end="/r")
            else:
                continue
    return [p.tables], np.array(p.tables)
def get_cubes_info(cube): #get Image mid-time, Sampling Mode, Exposure, Distance, Mean Resolution, Sub-spacecraft point, subsolar point, incidence, emergence, phase, and limb visible
    cube_info = work_parse_for_the_info_tables(cube)
    cub = cube_info[1]
    cu = [cub[index] + cub[index+1] for index in range(0,len(cub),2)]
    cube_info = [cube_info[0],cu]
    return cube_info
def analyze_folder_for_cube_info(folder_path,url):
    results = []
    # iterate through all files in the folder
    a = os.walk(folder_path)
    walked = [os.path.join(dp, f).replace("\\","/") for dp, dn, fn in os.walk(folder_path) for f in fn if any((f.endswith(".jpg"),f.endswith(".png"), f.endswith(".tif")))]
    length = len(walked)
    start_time = time.time()
    cube = [url + os.path.splitext(filename.split('/')[-1])[0] for filename in walked]
    cube_info = get_cubes_info(cube)

    return cube_info

def analyze_folder_for_visibility(folder_path,url):
    results = []
    # iterate through all files in the folder
    a = os.walk(folder_path)
    walked = [os.path.join(dp, f).replace("\\","/") for dp, dn, fn in os.walk(folder_path) for f in fn if any((f.endswith(".jpg"),f.endswith(".png"), f.endswith(".tif")))]
    length = len(walked)
    start_time = time.time()
    file_sizes = [get_file_size(file) for file in walked]
    curr_size = 0
    total_size = sum(file_sizes)
    cube = [url + os.path.splitext(filename.split('/')[-1])[0] for filename in walked]
    cube_info = get_cubes_info(cube)
    for index, filename in enumerate(walked):
        image_path = filename
        image, x,y,radius, match_percentage = analyze_image_for_visibility(image_path)
        curr_cube = [filename, image, x,y,radius, match_percentage]
        curr_cube.extend(cube_info[index])
        results.append([filename, image, x,y,radius, match_percentage])
        curr_size += file_sizes[index]
        expected_time_function(start_time, index, length, curr_size, total_size)
    return
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
        cube = filename.replace("//", "/").split('/')[-1]
        flyby = [f for f in filename.replace("//", "/").split('/') if "TI" in f][0]
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


if __name__ == "__main__":  
    baseurl = 'https://vims.univ-nantes.fr'
    searchurl = 'https://vims.univ-nantes.fr/target/titan'
    csv_save_path = 'nantes.csv'
    urls = parse_nantes_for_all_flybys(searchurl)
    #get all the nantes data for each flyby
    if os.path.exists("code/VIMS Data Ingestion/data/flybys.pickle"):
        with open("code/VIMS Data Ingestion/data/flybys.pickle", "rb") as f:
            flyby= pickle.load(f)

    else:
        with open("code/VIMS Data Ingestion/data/flybys.pickle", "wb") as f:
            # Use pickle to dump the variable into the file
            flyby = parse_for_the_info_tables(urls)
            pickle.dump(flyby, f)
    flybyarray = flyby[1]
    flybystuff = flyby[0]
    flybydict = combine_table_with_url(flybyarray, urls)
    if os.path.exists("code/VIMS Data Ingestion/data/nantes.csv"):
        nantes_csv = np.array(pd.read_csv('code/VIMS Data Ingestion/data/nantes.csv'), dtype = 'O')
    else:
        write_flyby_data(flybydict, "code/VIMS Data Ingestion/data/nantes.csv")
    
    ##
    ## NANTES DATA FOR FLYBYS PUT IN CSV
    ##
    
    parent_folder = "data/Nantes"
    # # call the function on the parent folder to get the list of results
    if os.path.exists("code/VIMS Data Ingestion/data/cubes.pickle"):
        with open("code/VIMS Data Ingestion/data/cubes.pickle", "rb") as f:
            results= pickle.load(f)

    else:
        with open("code/VIMS Data Ingestion/data/cubes.pickle", "wb") as f:
            # Use pickle to dump the variable into the file
            results = analyze_folder_for_visibility(parent_folder)
            pickle.dump(results, f)

    ##
    ##GET CUBE INFO
    ##
    
    new_url = "https://vims.univ-nantes.fr/cube/"

    if os.path.exists("code/VIMS Data Ingestion/data/cube_info.pickle"):
        with open("code/VIMS Data Ingestion/data/cube_info.pickle", "rb") as f:
            results= pickle.load(f)

    else:
        results = analyze_folder_for_cube_info(parent_folder, new_url)
        with open("code/VIMS Data Ingestion/data/cube_info.pickle", "wb") as f:
            # Use pickle to dump the variable into the file
            pickle.dump(results, f)
    
    show_positive_results(results) #show limb photos
    write_results(results, nantes_csv, "data/flyby_info/nantes_cubes.csv")


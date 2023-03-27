import pickle
import requests
import collections
collections.Callable = collections.abc.Callable
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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import Pool

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

def format_string(string):#function that gets rid of the extra spaces and /n's
    string = string.replace("\n", "").replace("","")
    new_string = string.replace("  ", " ").replace("|", " | ").strip()
    new_string = " | ".join([x.strip() for x in new_string.split("|")])
    checker = new_string.split(" ")
    if len(checker) == 2 and checker[0].isdigit() and checker[1].isdigit():
        new_string = 'x'.join(checker)
    return new_string
def read_a_url(url):
    req = urllib.request.Request(url=url, headers={'User-Agent': 'Chrome/111.0.5563.111'})
    index = 0
    while True:
        if index > 5:
            return None
        try:
            f = urllib.request.urlopen(req)
            break
        except:
            print("stuck at",url)
            index+=1
    # reading contents of the website
    return f.read()
def combine_table_with_url(flybyarray, urls) -> dict:
    return {url :flybyarray[index] for index, url in enumerate(urls)} 

def table_parser(urllist):
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
def parse_url_for_filenames(url):
    contents = read_a_url(url).decode('utf-8')
    start = 0
    parsed_url = urllib.parse.urlparse(url)
    parsed_url = url[0:url.index(parsed_url.netloc)] + parsed_url.netloc
    # domain = '{uri.netloc}/'.format(uri=parsed_uri)
    ret_url = []
    flyby = url.split('/')[-1]
    while True:
        finded = contents.find("/cube/",start)
        if finded > -1:
            founded =  contents.find("/cube/", finded+20)
            content_str = contents[finded:founded]
            content = content_str.split("\n")
            url_ = content[0].split('" title')[0]
            src = content[3].strip().replace("src=","").replace('"','')
            if "data/previews" in src:
                ret_url.append(src)
            start = founded
        else:
            break
    return_filename = [flyby + "/" + ret.split('/')[-1] for ret in ret_url]
    url = [(parsed_url + '/' + '/'.join(ret[1::].split('xxx'))) for ret in ret_url]
    return return_filename, url
def cube_info_parser(urllist):
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
            if len(p.tables) % 2 != 0 :
                pause = 0 #
    return [p.tables], np.array(p.tables)
def nantes_cubes_information(cube): #get Image mid-time, Sampling Mode, Exposure, Distance, Mean Resolution, Sub-spacecraft point, subsolar point, incidence, emergence, phase, and limb visible
    cube_info = cube_info_parser(cube)
    cub = cube_info[1]
    return cube_info

def download_image(url, filename=None):
    if filename is None:
        filename = url
    if not os.path.exists('/'.join(filename.split('/')[0:-1])):
        os.mkdir('/'.join(filename.split('/')[0:-1]))
    if not os.path.exists(filename):
        try:
            urllib.request.urlretrieve(url, filename)
        except:
            print(url, "did not work")
def analyze_image_for_visibility(image_path):
    # read in the image
    img = cv2.imread(image_path)
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.blur(gray, (9, 9))
    gray_blurred = cv2.threshold(gray_blurred,50,225, cv2.THRESH_BINARY)[1]
    # detect circles in the image
    circles = cv2.HoughCircles(gray_blurred, 
        method = cv2.HOUGH_GRADIENT_ALT, dp = 1,minDist = 10, param1 = 60,
        param2 = 0.7, minRadius= int(np.sqrt(img.shape[0]*img.shape[1]) / 10), 
        maxRadius = int(np.mean((img.shape[0], img.shape[1]))/1.5))
    percentage = 1.0
    if circles is None:
        contours, hierarchy = cv2.findContours(gray_blurred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = list(contours)
        if len(contours) > 1:
            contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
            g = np.zeros(gray_blurred.shape)
            g = cv2.fillConvexPoly(g, contour, 255)
            (x,y),radius = cv2.minEnclosingCircle(contour)
            percentage = (np.sum(g)/255) / (radius**2*np.pi)
        elif len(contours) == 0:
            percentage = 0.0
            x = -1
            y = -1
            radius = -1
        else:
            contour = contours[0]
            g = np.zeros(gray_blurred.shape)
            g = cv2.fillConvexPoly(g, contour, 255)
            (x,y),radius = cv2.minEnclosingCircle(contour)
            percentage = (np.sum(g)/255) / (radius**2*np.pi)
    elif circles is not None:
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
            if percentage >= 1: percentage = 1/percentage
    print(radius,percentage)

    # cv2.imshow(image_path.replace("\\","/").split("/")[-1] + " " + str(percentage), cv2.resize(gray_blurred,(gray_blurred.shape[1]*3,gray_blurred.shape[0]*3)))
    # cv2.waitKey(500)
    # if cv2.waitKey(1) == ord('e'):
    #     x = 9
    return img, x, y, radius, percentage

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
def write_results(results, filepath):
    with open(filepath, 'w', newline="") as f:
        # create the csv writer
        writer = csv.writer(f)
        # write a row to the csv file
        for row in results:
            writer.writerow(row)
    print("rows written")
def get_or_save_pickle(file,data = None): 
    if os.path.exists(file):
        with open(file, "rb") as f:
            data = pickle.load(f)
    else:
        if data == None:
            raise ValueError("data must be provided if file does not exist")
        with open(file, "wb") as f:
            # Use pickle to dump the variable into the file
            pickle.dump(data, f)    
    return data

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
def save_and_download_image(urls, filenames, basepath):
    leng = len(urls)
    for index, url in enumerate(urls):
        if filenames[index][0] == "/":
            path = basepath + "/" + filenames[index][1::]
        else:
            path = basepath + "/" + filenames[index]
        print(index,  "/", leng, "  ", url, "downloaded")
        download_image(url,path )
def analyze_folder_for_visibility(folder_path):
    results = []
    # iterate through all files in the folder
    a = os.walk(folder_path)
    walked = [os.path.join(dp, f).replace("//","/") for dp, dn, fn in os.walk(folder_path) for f in fn if any((f.endswith(".jpg"),f.endswith(".png"), f.endswith(".tif")))]
    length = len(walked)
    start_time = time.time()
    file_sizes = [get_file_size(file) for file in walked]
    curr_size = 0
    total_size = sum(file_sizes)
    for index, filename in enumerate(walked):
        image_path = filename
        image, x,y,radius, match_percentage = analyze_image_for_visibility(image_path)
        results.append([filename, image, x,y,radius, match_percentage])
        curr_size += file_sizes[index]
        expected_time_function(start_time, index, length, curr_size, total_size)
    return results
def get_all_nantes_cubes_information(folder_path,url):
    results = []
    # iterate through all files in the folder
    a = os.walk(folder_path)
    walked = [os.path.join(dp, f).replace(folder_path,"").replace("\\","/") for dp, dn, fn in os.walk(folder_path) for f in fn if any((f.endswith(".jpg"),f.endswith(".png"), f.endswith(".tif")))]
    
    length = len(walked)
    start_time = time.time()
    cube = [(url + os.path.splitext(filename.split('/')[-1])[0]) for filename in walked]
    cube_info = nantes_cubes_information(cube)

    return cube_info
def fix_flyby_with_multiple_targets(flyby):
    # a = [(index,row[1][1].count("|")) for index,row in enumerate(flyby) if row[1][1] != "Titan" and row[0][0] == "Name"]
    tester = [fly for fly in flyby if "Titan" in fly[1][1] or "TI" in fly[0][1].upper()]
    ret = [tester[index] + tester[index+1] for index in range(0,len(tester),2)]
    return ret
def combine_cube_info_and_vis(cube_information,visibility):
    visbility = []
    cube_indexes = [x[0][1] for x in cube_information]
    flyby_index= [index for index,c in enumerate(cube_information[0]) if "flyby" in c[0].lower()][0]
    for index,cuber in enumerate(visibility):
        file_loc = os.path.splitext(cuber[0].replace("\\","/").split('/')[-1])[0]
        cube = cuber[2::]            

        try:
            data = cube_information[cube_indexes.index(file_loc)]
            data = [format_string(dat[1]) for dat in data]
            data.extend(cube)
        except:
            data = [file_loc] + ["_"]*(len(cube_information[0])-1)
            data[flyby_index] = os.path.splitext(cuber[0].replace("\\","/").split('/')[-2])[0]

            data.extend(cube)
        visbility.append(data)
        print(index, "/", len(visibility), "images analyzed")
    headers= [x[0] for x in cube_information[0]]
    headers.extend(["x", "y", "radius", "percentage"])
    return visbility, headers
def show_positive_results(results):
    for i in results:
        filename, image, x,y,radius, match_percentage = i
        if match_percentage == 1:
            cv2.imshow("image",image)
            cv2.waitKey(100)
if __name__ == "__main__":
    """Get all flyby urls, then get all cubes, then analyze and log the cubes""" 
    searchurl = 'https://vims.univ-nantes.fr/target/titan'
    folder = 'D:/Nantes'
    
    #### START
    urls = parse_nantes_for_all_flybys(searchurl)
    #get all the nantes data for each flyby
    try:        
        flyby = get_or_save_pickle("code/VIMS Data Ingestion/data/flyby_info.pickle")
    except ValueError:
        flyby = table_parser(urls)
        get_or_save_pickle("code/VIMS Data Ingestion/data/flyby_info.pickle", flyby)
    flybyarray = flyby[1]
    flybystuff = flyby[0]
    flybydict = combine_table_with_url(flybyarray, urls)
    
    if os.path.exists("code/VIMS Data Ingestion/data/nantes_flybys.csv"):
        nantes_csv = np.array(pd.read_csv('code/VIMS Data Ingestion/data/nantes_flybys.csv'))
    else:
        write_flyby_data(flybydict, "code/VIMS Data Ingestion/data/nantes_flybys.csv")
    print("found all flybys")

    ##
    ## NANTES DATA FOR FLYBYS PUT IN CSV
    ##

    try:
        url,filename = get_or_save_pickle("code/VIMS Data Ingestion/data/all_files.pickle")
    except:
        filename = []; url = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for targeted_flyby in nantes_csv:
                ab, bc = parse_url_for_filenames(targeted_flyby[0])
                filename.extend(ab)
                url.extend(bc)
                print("completed", targeted_flyby, len(ab))  
        get_or_save_pickle("code/VIMS Data Ingestion/data/all_files.pickle", (url,filename))
        save_and_download_image(url,filename, "D:/Nantes")
    print("all images downloaded")
    
    ##
    ## GET VISIBILITY
    ##
    
    try:        
        vis = get_or_save_pickle("code/VIMS Data Ingestion/data/all_cubes.pickle")
    except ValueError:
        vis = analyze_folder_for_visibility(folder)
        get_or_save_pickle("code/VIMS Data Ingestion/data/all_cubes.pickle", vis)
    print("got cube visibility")
    
    ##
    ## GET CUBE INFO
    ##
    
    new_url = "https://vims.univ-nantes.fr/cube/"
    try:
        results = get_or_save_pickle("code/VIMS Data Ingestion/data/parsed_info.pickle")
    except:
        results= fix_flyby_with_multiple_targets(get_all_nantes_cubes_information(folder, new_url)[1])
        get_or_save_pickle("code/VIMS Data Ingestion/data/parsed_info.pickle", results)
    print("got information for all cubes")

    try:
        array = get_or_save_pickle("code/VIMS Data Ingestion/data/combined_nantes.pickle")
    except:
        arr = combine_cube_info_and_vis(results,vis)
        array = arr[0]
        array.insert(0,arr[1])
        get_or_save_pickle("code/VIMS Data Ingestion/data/combined_nantes.pickle", array)
    print("combined derived visibility and cube info")
    
    write_results(array, "code/VIMS Data Ingestion/data/combined_nantes.csv")
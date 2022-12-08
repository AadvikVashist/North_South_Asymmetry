import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import numpy as np
import time
import urllib.request
from pprint import pprint
from html_table_parser.parser import HTMLTableParser
from urllib.parse import urljoin
import urllib.parse
import csv
nantes_csv = pd.read_csv('C:/Users/aadvi/Desktop/North_South_Asymmetry/data/flyby_info/nantes.csv')
nantes_csv = np.array(nantes_csv)
folder = 'C:/Users/aadvi/Desktop/North_South_Asymmetry/data/flyby_info/'

targeted_flybys = [row for row in list(nantes_csv) if "|" in row[1]]
non_targeted_flybys = [row for row in list(nantes_csv) if "|" not in row[1]]
nantes_csv = nantes_csv

def url_get_contents(url):
    req = urllib.request.Request(url=url)
    f = urllib.request.urlopen(req)
    # reading contents of the website
    return f.read()
def parse_url(url):
    contents = url_get_contents(url).decode('utf-8')
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
            if "missing-vis" not in content_str:
                content = [cont for cont in content if "src" in cont][0]
                content = content.replace("src=",""); content = content.replace('"','')
                content = content.strip()
                ret_url.append(content)
            start = founded
        else:
            break
    ret_url = ret_url[0:-2]
    return_filename = [flyby + "/" + ret.split('/')[-1] for ret in ret_url]
    url = [(parsed_url + '/' + '/'.join(ret[1::].split('xxx'))) for ret in ret_url]
    return return_filename, url
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
def save_and_download(urls, filenames, basepath):
    leng = len(urls)
    for index, url in enumerate(urls):
        if filenames[index][0] == "/":
            path = basepath + "/" + filenames[index][1::]
        else:
            path = basepath + "/" + filenames[index]
        print(index,  "/", leng, "  ", url, "downloaded")
        download_image(url,path )
# download the file and save it to a local file
filename = []
url = []
for targeted_flyby in non_targeted_flybys:
    ab,bc = parse_url(targeted_flyby[0])
    filename.extend(ab)
    url.extend(bc) 
    print("completed", targeted_flyby)
save_and_download(url,filename, "C:/Users/aadvi/Desktop/Base")
filename = []
url = []
for targeted_flyby in targeted_flybys:
    ab,bc = parse_url(targeted_flyby[0])
    filename.extend(ab)
    url.extend(bc) 
    print("completed", targeted_flyby)
save_and_download(url,filename, "C:/Users/aadvi/Desktop/Base")

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
import csv
# for converting the parsed data in a
# pandas dataframe
import pandas as pd
baseurl = 'https://vims.univ-nantes.fr'
searchurl = 'https://vims.univ-nantes.fr/target/titan'
csv_save_path = 'nantes.csv'
def parse_directory_for_links(search_directory):
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
urls = parse_directory_for_links(searchurl)
def url_get_contents(url):
    req = urllib.request.Request(url=url)
    f = urllib.request.urlopen(req)
    # reading contents of the website
    return f.read()
def list_all_tables(urllist):
    p = HTMLTableParser()
    start = time.time()
    length = len(urllist)
    for index,url in enumerate(urllist):
        html = url_get_contents(url).decode('utf-8')
        p.feed(html)
        end = time.time() - start
        print(url, "                                                                                            ")
        print(url, "passed time",str(end) + "/" +str(length/(index+1)*end) + " seconds             ", end = "\r")
    return [p.tables], np.array(p.tables)
flybylist, flybyarray = list_all_tables(urls)
def combine_table_with_url(flybyarray, urls):
    return {url :flybyarray[index] for index, url in enumerate(urls)} 
flybyarrays = combine_table_with_url(flybyarray, urls)
print(flybyarray)
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
    for index, flyby in enumerate(flybyarray):
        writer.writerow([urls[index], flyby[indexOfFlybyIndex][1], flyby[indexOfCassiniMission][1], flyby[infexOfFlybyDate][1],flyby[indexOfAltitude][1],flyby[indexOfCubeQuantity][1],flyby[indexOfSequence][1],flyby[indexOfRevolution][1]])



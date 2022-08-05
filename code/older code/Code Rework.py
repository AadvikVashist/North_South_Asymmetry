#iterate through directory
import os
import os.path
from re import X
import statistics
#other
import sys
import time

#Regressions and Plotting
import math
import matlab
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab
import scipy
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
#image analysis
from PIL import Image, ImageStat
from scipy import stats
from scipy.optimize import curve_fit
from scipy.stats import mode, norm  
def Jcube_to_csv(nL, nS, csv=None, geo_csv=None, bands=None, var=None):
    '''Converts any Jcube or geocube csv file to numpy 2D array
    Note: Requires pandas and numpy.
    
    Parameters
    ----------
    csv: str
        Name of Jcube csv file with raw I/F data 
    geo_csv: str
        Name of csv file from Jason's geocubes with geographic info on a VIMS cube (.cub) 
        Can include relative directory location (e.g. 'some_dir/test.csv')
    nL: int
        number of lines or y-dimension of VIMS cube
    nS: int
        number of samples or x-dimension of VIMS cube
    vars: str
        geo_cube parameters that include "I/F", lat", "lon", "lat_res", "lon_res", 
        "phase", "inc", "eme", and "azimuth"
    '''
    ## check var is the correct string
    if var not in ["lat", "lon", "lat_res", "lon_res", "phase", "inc", "eme", "azimuth"] and var is not None:
        raise ValueError("Variable string ('var') is not formatted as one of the following options: \n"+\
                        '"I/F", "lat", "lon", "lat_res", "lon_res", "phase", "inc", "eme", "azimuth"')
    
    ## create image of Titan
    if geo_csv is None:
        ## read csv; can include relative directory
        csv = pd.read_csv(csv, header=None)
    
        def getMeanImg(csv, bands, nL, nS):
            '''Get the mean I/F image for a set of wavelengths from a Jcube csv file
            Note: a single 'bands' value will return the image at that wavelength.
            
            Parameters
            ----------
            csv: str 
                name of csv file for VIMS cube
            bands: int or list of int 
                band values from 96-352 for near-infrared VIMS windows
            nL: int
                number of lines 
            nS: int
                number of samples
            '''
            if isinstance(bands, int):
                bands = [bands]
            img = []
            for band in bands:
                cube = np.array(csv)[:,band].reshape(nL,nS)
                cube[cube < -1e3] = 0
                img.append(cube)#[band, :, :])
            return np.nanmean(img, axis=0)
        return getMeanImg(csv=csv, bands=bands, nL=nL, nS=nS)
    
    ## create geocube 
    if csv is None:
        ## read csv; can include relative directory
        geo = pd.read_csv(geo_csv, header=None)
        ## output chosen variable 2D array
        if var == 'lat':
            return np.array(geo)[:,0].reshape(nL,nS)
        if var == 'lon':
            return np.array(geo)[:,1].reshape(nL,nS)
        if var == 'lat_res':
            return np.array(geo)[:,2].reshape(nL,nS)
        if var == 'lon_res':
            return np.array(geo)[:,3].reshape(nL,nS)
        if var == 'phase':
            return np.array(geo)[:,4].reshape(nL,nS)
        if var == 'inc':
            return np.array(geo)[:,5].reshape(nL,nS)
        if var == 'eme':
            return np.array(geo)[:,6].reshape(nL,nS)
        if var == 'azimuth':
            return np.array(geo)[:,7].reshape(nL,nS)
    
    ## create geocube 
    if csv is None:
        ## read csv; can include relative directory
        geo = pd.read_csv(geo_csv, header=None)
        return (np.array(geo)[:,0].reshape(nL,nS), np.array(geo)[:,1].reshape(nL,nS))
def brightness( im_file ):
    im = Image.open(im_file).convert('L')
    stat = ImageStat.Stat(im)
    return stat.mean[0]
def polyfit(x, y, degree):
    results = {}
    coeffs = np.polyfit(x, y, degree)
    p = np.poly1d(coeffs)
    #calculate r-squared
    yhat = p(x)
    ybar = np.sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)
    sstot = np.sum((y - ybar)**2)
    return ssreg / sstot
def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / 4 / stddev)**2)
def poly6(x, g, h, i, j, a, b, c):
    return g*x**6+h*x**5+i*x**4+j*x**3+a*x**2+b*x+c
def poly6Prime(x, g, h, i, j, a, b, c):
    return 6*g*x**5+5*h*x**4+4*i*x**3+3*j*x**2+2*a*x+b
def running_mean(x, N):
    return np.convolve(x, np.ones(N)/N, mode='valid')
def analysis(dataset, imageDataset, csvFile, filename, crop, num_of_nans):
    fullFilename = filename
    try:
        im = plt.imread(fullFilename)[:,:,0]
    except:
        im = plt.imread(fullFilename)[:,:]
    csvFile = dataset + csvFile + os.listdir((dataset + csvFile))[0]
    height = len(im)
    width = len(im[0])
    lat = Jcube_to_csv(geo_csv = csvFile, var='lat', nL=height, nS=width)#nL=125, nS=133)
    lon = Jcube_to_csv(geo_csv = csvFile, var='lon', nL=height, nS=width)#nL=125, nS=133)
   
    im = im.astype(np.float32)
    ## create high-contrasting (HC) band image (I/F difference width)
    hc_band = np.empty((height, width), float)#125, 133), float)
    nsa_lats = []; nsa_lons = []; cols = []
    nans = [np.nan]*num_of_nans
    for col in range(crop[2], crop[3]):#im.shape[1]):
        shift_IF_down = np.insert(im[:,col], [0]*num_of_nans, nans)   #shift_IF_down  = np.insert(im[:,col], [0,0], [np.nan,np.nan])
        #print(im[:,col].flatten().size)#, np.insert(im[:,col], shift, nans))
        shift_IF_up = np.concatenate((im[:,col], nans))  #shift_IF_up  = np.concatenate((im[:,col],[np.nan],[np.nan]))
        subtraction = (shift_IF_up - shift_IF_down)
        hc_band[:,col] = (shift_IF_up - shift_IF_down)[int(num_of_nans/2):int(-num_of_nans/2)]
        #hc_band[crop[0]:crop[1],crop[2]:crop[3]]
        if_sh = hc_band[:,col][(lat[:,col] > -30.0) & (lat[:,col] < 0.0)] ## subset HC band column b/t 30°S to 0°N 
        lat_sh = lat[:,col][(lat[:,col] > -30.0) & (lat[:,col] < 0.0)]  ## subset HC band column b/t 30°S to 0°N 
        lon_sh = lon[:,col][(lat[:,col] > -30.0) & (lat[:,col] < 0.0)]  ## subset HC band column b/t 30°S to 0°N 
      

        popt, _ = curve_fit(poly6, lat_sh, if_sh)  ## apply 6th-order polyfit to I/F brightness profile (poly6 is function used for curved fit, lat = x, if = y)
        lat_sh_val = poly6(lat_sh, *popt) == np.min(poly6(lat_sh, *popt))
        nsa_lats.append((lat_sh[lat_sh_val][0])) ## find and record latitude of minimum I/F in 6th-order polyfit
    
        cols.append(col)
        plt.figure(figsize =(8,8))
        plt.plot(lat_sh, if_sh, label = 'brightness')
        plt.xlabel('latitude');plt.ylabel('I/F')
        plt.plot(lat_sh, poly6(lat_sh, *popt), label = 'polynomial fit')
        plt.plot(lat_sh, poly6Prime(lat_sh, *popt), label = 'polynomial fit derivative')
        plt.scatter(nsa_lats[-1], poly6(lat_sh, *popt)[lat_sh_val], label = 'NSA' )
        plt.plot([-30,-20,-10, 0], [0,0,0,0], label = 'y = 0')
        plt.legend()
        plt.show()
    deviation = np.std(nsa_lats)
    average = np.mean(nsa_lats)
    movingAverageList = running_mean(nsa_lats, 5)
    movingAverage =np.mean(movingAverageList)
    print(average, movingAverage)
   
    
"""def fileWrite(dataset, ):
    x = dataset + "Results/"
    if anomaly == False:
        D = open(x + "/Anomaly.txt","a+")
        D.write(filename)
        D.close()
    else:
        A = open(x + "/Lat.txt","a+")
        B = open(x + "/Deviations.txt","a+")
        C = open(x + "/Filenames.txt","a+")
        A.write(str(nsaLatW) + "\n")
        B.write(str(deviation) + "\n")
        C.write( filename + "\n")
        A.close()
        B.close()
        C.close()    """
def main(dataset, imageDataset, csvData,crop, nans):
    fullFilename = os.listdir(dataset + imageDataset)
    fullFilename = [dataset + imageDataset + e for e in fullFilename]
    #determine file types
    #full file names 
    acceptedImages = list(range(16,26)) +  list(range(70,77))+ list(range(84,92)) #list(range(16,26)) + 
    #Make folder if folder does not already exist
    folderPath = dataset+"Results/"
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)
    #rewrite files
    latFile = open(folderPath + "/Lat.txt","w")
    deviationFile = open(folderPath + "/Deviations.txt","w")
    anomalyFile = open(folderPath + "/Filenames.txt","w")
    filenameFile = open(folderPath + "/Anomaly.txt","w")
    #Image Analysis                                
    for filename in fullFilename:
        x = filename.split('.')
        x = [i for i in x if i.isdecimal()]
        try:
            x = int(x[-1])
        except:
            continue
        if x in acceptedImages:
            analysis(dataset, imageDataset, csvData, filename, crop,nans)
    #image file types accepted
dataset1= main('C:/Users/aadvi/Desktop/Titan Paper/Data/T62/', 'T62.vis.cyl/', 'T62.CSV/',[0, 125, 0, 133], 2)

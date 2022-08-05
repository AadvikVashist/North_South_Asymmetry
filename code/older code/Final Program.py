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
import cv2; from pyvims.projections import equi_cube

def Jcube_to_csv(nL, nS, csv=None, geo_csv=None, bands=None, var=None): #Get image to create arrays that help read image data as latitude data
    '''Converts any Jcube or geocube csv file to numpy 2D array
    Note: Requires pandas and numpy.
    
    Parameters
    ----------
    csv: str
        Name of Jcube csv file with raw I/F data 
    geo_csv: str
        Name of csv file from geocubes with geographic info on a VIMS cube (.cub) 
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
def polyfit(x, y, degree): #alternate fit for polynomials
    results = {}
    coeffs = np.polyfit(x, y, degree)
    p = np.poly1d(coeffs)
    #calculate r-squared
    yhat = p(x)
    ybar = np.sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)
    sstot = np.sum((y - ybar)**2)
    return ssreg / sstot
def gaussian(x, amplitude, mean, stddev): #gaussian fit
    return amplitude * np.exp(-((x - mean) / 4 / stddev)**2)
def poly6(x, g, h, i, j, a, b, c): #sextic function
    return g*x**6+h*x**5+i*x**4+j*x**3+a*x**2+b*x+c
def poly6Prime(x, g, h, i, j, a, b, c): #derivative of sextic function
    return 6*g*x**5+5*h*x**4+4*i*x**3+3*j*x**2+2*a*x+b
def poly6Derivative(g, h, i, j, a, b, c): #derivative of sextic function coefficents
    return [6*g,5*h,4*i,3*j,2*a,b]
def running_mean(x, N): #window avearage
    return np.convolve(x, np.ones(N)/N, mode='valid')
def brightnessDifference(lat, im, nsaLat):
    splitY = min(range(len(lat[:, 0])), key=lambda x:abs(lat[x,0]-nsaLat))
    southSplit = min(range(len(lat[:, 0])), key=lambda x:abs(lat[x,0]-30))
    northSplit = min(range(len(lat[:, 0])), key=lambda x:abs(lat[x,0]+30))
    north = im[splitY:northSplit, :]
    south = im[southSplit:(splitY+1), :]
    northM = np.mean(north)
    southM = np.mean(south)
    return northM/southM
def analysis(dataset, imageDataset, csvFile, filename, crop, num_of_nans, goalNSA):
    #open image arrays
    try:
        im = plt.imread(filename)[:,:,0]
    except:
        im = plt.imread(filename)[:,:]
    height = len(im)
    width = len(im[0])   
    if crop[1] == 0 and crop[0] == 0:
        crop[1] = width
    if crop[2] == 0 and crop[3] == 0:
        crop[3] = height
    im = im.astype(np.float32)
    hc_band = np.empty((height, width), float)
    nsa_lats = []; nsa_lons = []; cols = []
    nans = [np.nan]*(int(num_of_nans))
    count = 0
    #get latitude values of each pixel using CSV
    lat = Jcube_to_csv(geo_csv = csvFile, var='lat', nL=height, nS=width)
    lon = Jcube_to_csv(geo_csv = csvFile, var='lon', nL=height, nS=width)

    for col in range(crop[0], crop[1]):
        shift_IF_down = np.insert(im[:,col], [0]*num_of_nans, nans) 
        shift_IF_up = np.concatenate((im[:,col], nans))
        subtraction = (shift_IF_up - shift_IF_down)
        hc_band[:,col] = subtraction[int(num_of_nans/2):int(-num_of_nans/2)]
        #hc_band[crop[0]:crop[1],crop[2]:crop[3]]
        if_sh = hc_band[:,col][(lat[:,col] > -30.0) & (lat[:,col] < 0.0)] ## subset HC band column b/t 30°S to 0°N 
        lat_sh = lat[:,col][(lat[:,col] > -30.0) & (lat[:,col] < 0.0)]  ## subset HC band column b/t 30°S to 0°N 
        lon_sh = lon[:,col][(lat[:,col] > -30.0) & (lat[:,col] < 0.0)]  ## subset HC band column b/t 30°S to 0°N 
        try:
            popt, _ = curve_fit(poly6, lat_sh, if_sh)#apply sextic regression to data
            sixFit = poly6(lat_sh, *popt)
            nsa = np.max(np.abs(sixFit))
            lat_sh_val = (sixFit == nsa)
            nsa_lats.append(lat_sh[lat_sh_val][0])#append to nsalats
        except:
            count+= 1
            poptD = poly6Derivative(*popt)#get derivative of sextic regression
            derivativeRoots = np.roots(poptD) #roots (Real and imaginary) of derivative function
            derivativeRoots = derivativeRoots[np.isreal(derivativeRoots)] #remove extraneous soulutions (imaginary)
            drIndex = min(range(len(derivativeRoots)), key=lambda x: abs(derivativeRoots[x]-goalNSA)) #find value closest to NSA
            derivativeRoots = derivativeRoots[drIndex] #get lat of derivative root
            nsa_lats.append(derivativeRoots.real) #append to nsa_lats
        if col > 301:
            print(nsa_lats[col])
    deviation = np.std(nsa_lats) #standard deviation
    average = np.mean(nsa_lats) #standard average
    movingAverageList = running_mean(nsa_lats, 2) #moving average
    movingAverage =np.mean(movingAverageList) #moving average
    diff = brightnessDifference(lat, im, movingAverage) #difference between north and south
    return movingAverage, deviation, diff, lat_sh
def fileWrite(avg, dev, diff, folderPath, setName, number):
    A = open(folderPath + setName + "_analysis.txt","a+")
    A.write(str(avg) +" " + str(dev) + " " + str(diff) + " " + str(number)  + "\n")
    A.close()
def main(setName, dataset, crop, nans, goalNSA):
    visCyl = 'vis.cyl/'
    csvData = 'csv'
    csvFile = dataset +  setName + '/' + csvData + '/' + os.listdir((dataset + setName + '/' + csvData))[0]
    fullFilename = os.listdir(dataset+ setName + '/' + visCyl)
    fullFilename = [dataset + setName + '/' + visCyl + e for e in os.listdir(dataset+ setName + '/' + visCyl)]
    imageFolder = dataset+visCyl
    acceptedImages = list(range(16,26)) +  list(range(70,77))+ list(range(84,92))
    #Make folder if folder does not already exist
    folderPath = dataset+"Results/"
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)
    #rewrite files
    open(folderPath + setName + "_analysis.txt","w")
    #Image Analysis
    fileWrite('NSA LAT  ', 'Deviation  ', 'N/S  ',  folderPath, setName, 'Image Number')
    runTime = []; avgs = []; devs = []; diffs = []                           
    for filename in fullFilename:
        start_time = time.time()
        #get file number in 0-95
        x = filename.replace('.tif', '') 
        x = x.replace(dataset+imageFolder, '')
        x = x.replace(setName, '')
        x = [i for i in x if i.isdecimal()]
        x = ''.join(str(e) for e in x)
        x = int(x)
        if x in acceptedImages: #check if x is accpeted file 
            avg, dev, diff, nsaLat = analysis(dataset, imageFolder, csvFile, filename, crop,nans, goalNSA)
            fileWrite(avg, dev, diff, folderPath, setName, x)
            avgs.append(avg)
            devs.append(dev)
            diffs.append(diff)
            runTime.append((time.time() - start_time))
    fileWrite(np.mean(avgs),  np.std(avgs), np.mean(diffs), folderPath, setName, (''.join(filter(str.isdigit, setName))).zfill(4)) #average for dataset
    os.startfile(folderPath + setName + "_analysis.txt") #open datset file
    print(np.mean(runTime))
    return setName,avg, dev, diff
#datasets
data = [['T85', 'C:/Users/aadvi/Desktop/Titan Paper/Data/', [0, 0, 0, 0], 6, -13]] #
for i in data:#iterate over lists
    dataset = main(*i)
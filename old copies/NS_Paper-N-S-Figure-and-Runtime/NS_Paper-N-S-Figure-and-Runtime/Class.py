#iterate through directory
from msilib.schema import Directory
import os
import os.path as path
from re import X
import statistics
from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename  
import tkinter as tk
import tkinter.filedialog as fd
#other
import sys
import time
import csv
#Regressions and Plotting
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
class tif:
    def __init__(self, dataset, dimension):
        root = tk.Tk()
        self.file = fd.askdirectory(parent=root, title='Choose root data folder') + '/'
        self.dataset = dataset
        self.nS = dimension[0]
        self.nL = dimension[1]
        self.visCylLocation = fd.askopenfilename(parent=root, title='Choose vis.cyl csv location')
        self.csvLocation = fd.askopenfilename(parent=root, title='Choose csv location')
        self.createDatasetFolder()
        self.createCSVFolder()
        self.createVisCylFolder()
        self.resultsFolder = self.datasetFolder + '/vis.cyl/'
        for currentBand in range(96):
            band = self.Jcube_to_csv(csv = self.visCylLocation, nL = self.nL, nS = self.nS, bands = currentBand)
            im = Image.fromarray(np.uint8(band * (255/np.max(band))) , 'L')
            im.save((self.resultsFolder + str(self.dataset) + "." + str(currentBand) + '.tif'))
    def createDatasetFolder(self):
        folderPath = self.file + self.dataset
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
        self.datasetFolder = folderPath
        if path.dirname(self.visCylLocation) != folderPath:
            string = (self.datasetFolder + self.visCylLocation.replace(path.dirname(self.visCylLocation),''))
            os.replace(self.visCylLocation, string)
            self.visCylLocation = string
    def createCSVFolder(self):
        folderPath = self.file + self.dataset + '/csv'
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
        self.resultsFolder = folderPath
        if path.dirname(self.csvLocation) != folderPath:
            string = (path.dirname(folderPath) + '/csv' + self.csvLocation.replace(path.dirname(self.csvLocation),''))
            os.replace(self.csvLocation, string)
            self.csvLocation = string
    def createVisCylFolder(self):
        folderPath = self.file + self.dataset + '/vis.cyl'
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
    def Jcube_to_csv(self, nL, nS, csv=None, geo_csv=None, bands=None, var=None): #Get image to create arrays that help read image data as latitude data
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
class data:
    def __init__(self, mainFolder, dataFolder, csvFolder,imageFolder, nans, goalNSA, datasetName):
        #set all self directory variables
        self.mainFolder = mainFolder
        self.datasetName = datasetName
        self.csvFolder = csvFolder
        self.imageFolder = imageFolder
        self.datasetLocation = mainFolder + "/" + datasetName
        self.imageFolderLoc = self.datasetLocation + "/" + self.imageFolder
        self.num_of_nans = nans
        self.goalNSA = goalNSA
        self.csvFile = self.datasetLocation + '/' + self.csvFolder + '/' + os.listdir((self.datasetLocation + '/' + self.csvFolder))[0]
        self.allFiles = [self.imageFolderLoc + '/' + e for e in os.listdir(self.imageFolderLoc)]
        #create folder and file
        self.createFolder()
        self.createFile()
        #create global variables for data
        self.NSA = []
        self.deviation = []
        self.IF = []
        self.NS = []
        self.lat = []
        self.lon = []
        self.iterations = []
        abcd = []
        for self.iteration in range(len(self.allFiles)):
            starts = time.time()
            self.currentFile = self.allFiles[self.iteration]
            self.iterations.append(self.iteration)
            self.analysis()
            print(self.NSA[self.iteration], self.deviation[self.iteration], self.NS[self.iteration], self.iteration)
            ends = time.time()
            print('Imagerun time:', ends- starts, '  Image ' , self.iteration )
            abcd.append(ends-starts)
        try:
            datasetAverage = self.averages([self.NSA, np.std(self.NSA), self.NS])
            self.NSA.append(datasetAverage[0]); self.deviation.append(datasetAverage[1]); self.NS.append(datasetAverage[2]); self.iterations.append(self.datasetName)
            self.NSA.insert(0, "NSA"); self.deviation.insert(0, "Deviation"); self.NS.insert(0,"N/S"); self.iterations.insert(0, "File Number")
            self.fileWrite(self.NSA)
            self.fileWrite(self.deviation)
            self.fileWrite(self.NS)
            self.fileWrite(self.iterations)
        finally:
            self.fileClose()
        print('Average', np.mean(abcd))

    def createFolder(self):
        folderPath = self.mainFolder+"/Result/"
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
        self.resultsFolder = folderPath
    def createFile(self):
        open(self.resultsFolder + self.datasetName + "_analytics.csv","w")
        x = open(self.resultsFolder + self.datasetName + "_analytics.csv","a+", newline='')
        self.analysisFile = x
    def fileWrite(self, x):
        a = csv.writer(self.analysisFile)
        a.writerow(x)
    def fileClose(self):
        self.analysisFile.close()
    def averages(self, x):
        X = []
        for i in range(len(x)):
            X.append(np.mean(x[i]))
        return X
    def Jcube_to_csv(self, nL, nS, csv=None, geo_csv=None, bands=None, var=None): #Get image to create arrays that help read image data as latitude data
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
    def brightness(self, im_file ):
        im = Image.open(im_file).convert('L')
        stat = ImageStat.Stat(im)
        return stat.mean[0]
    def polyfit(self, x, y, degree): #alternate fit for polynomials
        results = {}
        coeffs = np.polyfit(x, y, degree)
        p = np.poly1d(coeffs)
        #calculate r-squared
        yhat = p(x)
        ybar = np.sum(y)/len(y)
        ssreg = np.sum((yhat-ybar)**2)
        sstot = np.sum((y - ybar)**2)
        return ssreg / sstot
    def gaussian(self, x, amplitude, mean, stddev): #gaussian fit
        return amplitude * np.exp(-((x - mean) / 4 / stddev)**2)
    def poly6(self, x, g, h, i, j, a, b, c): #sextic function
        return g*x**6+h*x**5+i*x**4+j*x**3+a*x**2+b*x+c
    def poly6Prime(self, x, g, h, i, j, a, b, c): #derivative of sextic function
        return 6*g*x**5+5*h*x**4+4*i*x**3+3*j*x**2+2*a*x+b
    def poly6Derivative(self, g, h, i, j, a, b, c): #derivative of sextic function coefficents
        return [6*g,5*h,4*i,3*j,2*a,b]
    def running_mean(self, x, N): #window avearage
        return np.convolve(x, np.ones(N)/N, mode='valid')
    def visualizeBrightnessDifferenceSamplingArea(self, im, x, nsaLat):
        im[x[0]:x[1],x[3]] *= 1.25
        im[(x[1]+1):x[2],x[3]] *=  0.8
        plt.imshow(im, cmap = 'Greys')
        plt.show()
    def brightnessDifference(self, im, nsaLat):
        splitY = min(range(len(self.lat[:, 0])), key=lambda x:abs(self.lat[x,0]-nsaLat))
        horizontalSample= 4
        verticalSample = 60
        northSplit = min(range(len(self.lat[:, 0])), key=lambda x:abs(self.lat[x,0]-verticalSample-nsaLat))
        southSplit = min(range(len(self.lat[:, 0])), key=lambda x:abs(self.lat[x,0]+verticalSample-nsaLat))
        
        horizontalCrop = range(int(len(im[0])/horizontalSample),(horizontalSample-1)*int(len(im[0])/horizontalSample))
        north = im[northSplit:splitY,horizontalCrop]
        south = im[(splitY+1):southSplit, horizontalCrop]
        #self.visualizeBrightnessDifferenceSamplingArea(im,[northSplit, splitY, southSplit, horizontalCrop], nsaLat) 
        northM = np.mean(north[north != 0.])
        southM = np.mean(south[south != 0.])
        return northM/southM
    def analysis(self):
        #open image arrays
        
        try:
            im = plt.imread(self.currentFile)[:,:,0]
        except:
            im = plt.imread(self.currentFile)[:,:]
        height = len(im)
        width = len(im[0])   
        im = im.astype(np.float32)
        hc_band = np.empty((height, width), float)
        nsa_lats = []; nsa_lons = []; cols = []
        nans = [np.nan]*(int(self.num_of_nans))
        count = 0
        #get latitude values of each pixel using CSV
        if self.iteration == 0:
            self.lat = self.Jcube_to_csv(geo_csv = self.csvFile, var='lat', nL=height, nS=width)
            self.lon = self.Jcube_to_csv(geo_csv = self.csvFile, var='lon', nL=height, nS=width)
            self.columnLat = self.lat[:,0]
            self.subset = tuple([(self.columnLat > -30.0) & (self.columnLat < 0.0)])
            self.lat_sh = self.columnLat[self.subset] ## subset HC band b/t 30°S to 0°N 
        for col in range(width):
            subtraction = (np.insert(im[:,col], [0]*self.num_of_nans, nans) - np.concatenate((im[:,col], nans)))
            hc_band[:,col] = subtraction[int(self.num_of_nans/2):int(-self.num_of_nans/2)]
            #hc_band[crop[0]:crop[1],crop[2]:crop[3]]
            columnHC = hc_band[:,col]
            if_sh = columnHC[self.subset] ## subset HC band b/t 30°S to 0°N 
            #lat_sh = self.columnLat[self.subset]  ## subset HC band b/t 30°S to 0°N 
            #lon_sh = self.lon[:,col][self.subset]  ## subset HC band b/t 30°S to 0°N 
            try:
                popt, _ = curve_fit(self.poly6, self.lat_sh, if_sh)#apply sextic regression to data
                poptD = self.poly6Derivative(*popt)#get derivative of sextic regression
                derivativeRoot = np.roots(poptD) #roots (Real and imaginary) of derivative function
                derivativeRoots = derivativeRoot[np.isreal(derivativeRoot)] #remove extraneous soulutions (imaginary)
                drIndex = min(range(len(derivativeRoots)), key=lambda x: abs(derivativeRoots[x]-self.goalNSA)) #find value closest to NSA
                derivativeRoots = derivativeRoots[drIndex] #get lat of derivative root
                nsa_lats.append(derivativeRoots.real) #append to nsa_lats
            except:
                pass
            
        
        dev = np.std(nsa_lats) #standard deviation
        average = np.nanmean(nsa_lats) #standard average
        movingAverageList = self.running_mean(nsa_lats, 2) #moving average
        movingAvg = np.mean(movingAverageList) #moving average
        diff = self.brightnessDifference(im, movingAvg) #difference between north and south
        self.NSA.append(movingAvg)
        self.deviation.append(dev)
        self.NS.append(diff)
        self.IF.append(if_sh)
class Figure:
    def __init__(self, directory, Tdataset):
        self.allDatasets = []
        self.NSA =[]
        self.NS_Flux_Ratio = []
        self.directory = directory
        self.Tdataset = Tdataset
        self.wavelengths()
        for self.i in Tdataset:
            self.allDatasets.append((self.directory[0] + "/Result/" + self.i + self.directory[3]))
            self.datasetRead(self.allDatasets[-1])
        self.wavelengths()
        self.fluxDimensions = [0.3,1.1,0.5,1.8]
        self.NS_Flux()
    def wavelengths(self):
        self.wavelength = (np.array(pd.read_csv(self.directory[0] + '/' + self.directory[1] + '/' + self.directory[2], header = None)))[0]
    def datasetRead(self, x):
        self.data = np.array(pd.read_csv(x, header = None))
        self.NSA.append(((self.data[0])[1:-1]).astype(np.float64))
        self.NS_Flux_Ratio.append(((self.data[2])[1:-1]).astype(np.float64))
        

    def NS_Flux(self):
        plt.figure(figsize =(15,12))
        plt.xlim(self.fluxDimensions[0:2])
        plt.ylim(self.fluxDimensions[2:4])
        plt.xlabel('wavelength');plt.ylabel('N/S')
        for i in range(len(self.Tdataset)):
            plt.plot(self.wavelength, self.NS_Flux_Ratio[i], label = self.Tdataset[i])
        plt.legend()
        plt.show()
        
        
        
class Titan:
    def __init__(self,directory, datasets):
        self.directory = directory
        self.datasets = datasets
        self.allDatasets = [] 
        for i in range(len(self.datasets)):
            x = data(*self.directory,self.datasets[i])
            self.allDatasets.append(x)  
#Titan(['C:/Users/aadvi/Desktop/Titan Paper/Data', 'Titan Data', 'csv/', 'vis.cyl/', 4, -13], ['Ta', 'T8','T31','T61','T62','T67','T79','T85', 'T92','T108','T114'])
#Titan(['C:/Users/aadvi/Desktop/Titan Paper/Data', 'Titan Data', 'csv/', 'vis.cyl/', 4, -13], ['Ta'])
Figure(['C:/Users/aadvi/Desktop/Titan Paper/Data', 'Titan Data', 'wavelength.csv','_analytics.csv'], ['Ta', 'T8','T31','T61','T62','T67','T79','T85', 'T92','T108','T114'])

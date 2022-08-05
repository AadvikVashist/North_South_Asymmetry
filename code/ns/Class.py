#iterate through directory
from msilib.schema import Directory
from nntplib import GroupInfo
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
import math
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab
import scipy
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import r2_score
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
            band = self.Jcube(csv = self.visCylLocation, nL = self.nL, nS = self.nS, bands = currentBand)
            im = Image.fromarray(np.uint8(band * (255/np.max(band))) , 'L')
            if currentBand < 10:    
                saveBand = "0" + str(currentBand)
                im.save((self.resultsFolder + str(self.dataset) + "." + saveBand + '.tif'))
            else:
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
    def Jcube(self, nL, nS, csv=None, geo_csv=None, bands=None, var=None): #Get image to create arrays that help read image data as latitude data
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
    def __init__(self, directory, datasetName, shiftDegree , purpose):
        #create all class paths,directories, and variables
        self.createDirectories(directory, datasetName)
        self.createLists(purpose, datasetName)
        self.analysisPrep(shiftDegree)
        #gets purpose condition and executes individual operations based on purpose
        self.conditionals()
    def createDirectories(self, directory, dataset): #create all file directory paths
        #basic datasets
        self.directoryList = directory
        self.flyby = dataset[0]
        self.masterDirectory = self.directoryList[0]
        self.csvFolder = self.directoryList[2]
        self.tFolder = self.directoryList[0] + "/" + dataset[0]
        self.imageFolder = self.directoryList[0] + "/" + dataset[0] + "/" + self.directoryList[3]
        #finding files
        self.csvFile = self.tFolder + '/' + self.csvFolder + '/' + os.listdir((self.tFolder + '/' + self.csvFolder))[0]
        self.allFiles = [self.imageFolder + '/' + e for e in os.listdir(self.imageFolder)]
        self.resultsFolder = self.masterDirectory+ "/" + self.directoryList[7] + "/"
        if len(self.allFiles) != 96:
            print("missing 1+ files")
    def createLists(self, purpose, NSA): #create global variables for data
        self.NSA = []
        self.deviation = []
        self.IF = []
        self.NS = []
        self.lat = []
        self.lon = []
        self.iterations = []
        self.goalNSA = NSA[1]
        self.goalOutlier = NSA[2]
        self.purpose = purpose
        if self.purpose[0] == "tilt":
            self.band = []
        elif self.purpose[0] == "if_sh":
            self.if_sh = []
            self.latSh = []
        self.play = False
    def analysisPrep(self,shiftDegree):
        try:
            self.im = plt.imread(self.allFiles[0])[:,:,0]
        except:
            self.im = plt.imread(self.allFiles[0])[:,:]
        self.height = len(self.im)
        self.width = len(self.im[0])   
        self.im = self.im.astype(np.float32)
        self.lat = self.Jcube(geo_csv = self.csvFile, var='lat', nL=self.height, nS=self.width)
        self.lon = self.Jcube(geo_csv = self.csvFile, var='lon', nL=self.height, nS=self.width)
        self.columnLat = self.lat[:,0]
        self.subset = tuple([(self.columnLat > self.goalNSA -15) & (self.columnLat < self.goalNSA + 15.0)])
        self.lat_sh = self.columnLat[self.subset] ## subset HC band b/t 30°S to 0°N 
        latRange = np.max(self.lat)-np.min(self.lat)
        latTicks = len(self.lat)/latRange
        self.num_of_nans = int(latTicks*shiftDegree)
        self.nans = [np.nan]*(self.num_of_nans)
        if self.purpose[0] == "if_sh":
            self.ifSubset = tuple([(self.columnLat > -90.0) & (self.columnLat < 90.0)])
    def createFolder(self):
        folderPath = self.masterDirectory + "/" + self.directoryList[7] + "/"
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
        self.resultsFolder = folderPath
    def createFile(self):
        open(self.resultsFolder + self.flyby + "_analytics.csv","w")
        x = open(self.resultsFolder + self.flyby + "_analytics.csv","a+", newline='')
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
    def Jcube(self, nL, nS, csv=None, geo_csv=None, bands=None, var=None): #Get image to create arrays that help read image data as latitude data
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
    def conditionals(self):
        if self.purpose[0] == "data":
            print("data")
            self.getDataControl()
        elif self.purpose[0] == "tilt":
            print("tilt analysis")
            self.getTiltControl()
        elif self.purpose[0] == "if_sh":
            print("if_sh")
            self.getIf_shControl()
        else:
            print("data output type not understood; ", self.purpose[0], " not valid")
    def getDataControl(self):
        for self.iter in range(len(self.allFiles)):
            self.currentFile = self.allFiles[self.iter]
            self.iterations.append(self.iter)
            self.dataAnalysis()
            print("dataset", self.flyby, "     image", "%02d" % (self.iter),"     boundary",format(self.NSA[self.iter],'.15f') ,"      deviation", format(self.deviation[self.iter],'.15f'), "        N/S",format(self.NS[self.iter],'.10f') )
        if "datasetAverage" in self.purpose[1]:
            datasetAverage = self.averages([self.NSA, np.std(self.NSA), self.NS])
            print("\n\n\n\n\n\n\n\n\n\n\n")
            for i in range(5):
                print("dataset", self.flyby,"                   boundary",format(datasetAverage[0],'.15f') ,"      deviation", format(datasetAverage[1],'.15f'), "        N/S",format(datasetAverage[2],'.10f') )
                time.sleep(0.25)
        #write data
        if "write" in self.purpose[1]:
            #create folder and file
            self.createFolder()
            self.createFile()
            try:
                datasetAverage = self.averages([self.NSA, np.std(self.NSA), self.NS])
                self.NSA.append(datasetAverage[0]); self.deviation.append(datasetAverage[1]); self.NS.append(datasetAverage[2]); self.iterations.append(self.flyby)
                self.NSA.insert(0, "NSA"); self.deviation.insert(0, "Deviation"); self.NS.insert(0,"N/S"); self.iterations.insert(0, "File Number")
                self.fileWrite(self.NSA)
                self.fileWrite(self.deviation)
                self.fileWrite(self.NS)
                self.fileWrite(self.iterations)
            finally:
                self.fileClose()
    def getIf_shControl(self):
        if len(self.purpose[1]) == 0:
            for self.iter in range(len(self.allFiles)):
                self.currentFile = self.allFiles[self.iter]
                self.iterations.append(self.iter)
                self.ifAnalysis()
        else:
            for self.iter in range(len(self.allFiles)):
                if self.iter in self.purpose[1]:
                    self.currentFile = self.allFiles[self.iter]
                    self.iterations.append(self.iter)
                    self.if_sh.append(self.ifAnalysis())
    def getTiltControl(self):
        #go through each file
        for self.iter in range(len(self.allFiles)):
            self.currentFile = self.allFiles[self.iter]
            if self.iter in self.purpose[1]:
                self.tiltAnalysis()
        #write data
            #print(self.band)
            pass
    def visualizeBrightnessDifferenceSamplingArea(self, im, x, nsaLat):
        im[x[0]:x[1],x[3]] *= 2
        im[(x[1]+1):x[2],x[3]] *=  0.5
        plt.imshow(im, cmap = 'Greys')
        plt.show()
    def brightnessDifference(self, im, nsaLat):
        splitY = min(range(len(self.lat[:, 0])), key=lambda x:abs(self.lat[x,0]-nsaLat))
        horizontalSample= 4
        verticalSample = 30
        northSplit = min(range(len(self.lat[:, 0])), key=lambda x:abs(self.lat[x,0]-verticalSample-nsaLat))
        southSplit = min(range(len(self.lat[:, 0])), key=lambda x:abs(self.lat[x,0]+verticalSample-nsaLat))
        
        #horizontalCrop = range(int(len(im[0])/horizontalSample),(horizontalSample-1)*int(len(im[0])/horizontalSample))
        #north = im[northSplit:splitY,horizontalCrop]
        #south = im[(splitY+1):southSplit, horizontalCrop]
        #self.visualizeBrightnessDifferenceSamplingArea(im,[northSplit, splitY, southSplit, horizontalCrop], nsaLat) 
        north = im[northSplit:splitY,:]
        south = im[(splitY+1):southSplit, :]
        
        #self.visualizeBrightnessDifferenceSamplingArea(im,[northSplit, splitY, southSplit, [ 0,724]], nsaLat) 
        northM = np.mean(north[north != 0.])
        southM = np.mean(south[south != 0.])
        return northM/southM
    def dataAnalysis(self):
        try: #open image arrays
            self.im = plt.imread(self.currentFile)[:,:,0]
        except:
            self.im = plt.imread(self.currentFile)[:,:]
        self.im = self.im.astype(np.float32)
        hc_band = np.empty((self.height, self.width), float)
        nsa_lats = []; nsa_lons = []; cols = []
        #get latitude values of each pixel using CSV
        width = []
        showPlot = "y"
        for col in range(self.width):
            subtraction = (np.insert(self.im[:,col], [0]*self.num_of_nans, self.nans) - np.concatenate((self.im[:,col], self.nans)))
            hc_band[:,col] = subtraction[int(self.num_of_nans/2):int(-self.num_of_nans/2)]
            #hc_band[crop[0]:crop[1],crop[2]:crop[3]]
            columnHC = hc_band[:,col]
            if_sh = columnHC[self.subset] ## subset HC band b/t 30°S to 0°N 
            #lat_sh = self.columnLat[self.subset]  ## subset HC band b/t 30°S to 0°N 
            self.lon_sh = self.lon[:,col][self.subset]  ## subset HC band b/t 30°S to 0°N 
            if np.min(if_sh) != np.max(if_sh):
                try:
                    popt, _ = curve_fit(self.poly6, self.lat_sh, if_sh)#apply sextic regression to data
                    #print(np.mean(if_sh))
                    poptD = self.poly6Derivative(*popt)#get derivative of sextic regression
                    derivativeRoot = np.roots(poptD) #roots (Real and imaginary) of derivative function
                    realDerivativeRoots = derivativeRoot[np.isreal(derivativeRoot)] #remove extraneous soulutions (imaginary)
                    drIndex = min(range(len(realDerivativeRoots)), key=lambda x: abs(realDerivativeRoots[x]-self.goalNSA)) #find value closest to NSA
                    derivativeRoots = realDerivativeRoots[drIndex]
                    if abs(derivativeRoots.real-self.goalNSA) >= self.goalOutlier:
                        width.append(False)
                    else: 
                        nsa_lats.append(derivativeRoots.real)
                        width.append(True)
                    if self.play and abs(derivativeRoots.real - self.goalNSA) > self.purpose[2]: ##show bad columns
                        if not showPlot == False:
                            showPlot = input("continue? ")
                            if showPlot == "":
                                plt.figure(figsize =(8,8))
                                x = self.flyby + " Band: " + str(self.iter) + "      column " + str(col) + "/" + str(self.width)
                                plt.title(x)
                                plt.plot(self.lat_sh, if_sh, label = 'brightness')
                                plt.xlabel('latitude');plt.ylabel('I/F')
                                plt.plot(self.lat_sh, self.poly6(self.lat_sh, *popt), label = 'sextic regression')
                                plt.plot(self.lat_sh, self.poly6Prime(self.lat_sh, *popt), label = 'sextic derivative')  
                                plt.plot([-30,-20,-10, 0], [0,0,0,0], label = 'y = 0')
                                plt.scatter(derivativeRoots.real,0 , label = 'NSA' )
                                plt.legend()
                                plt.show(block=False)
                                plt.pause(2)
                                plt.close()
                            else:
                                showPlot = False
                except:
                    width.append(False)
                    pass
            else:
                width.append(False)
        if "showNSA" in self.purpose[1]:
            zipped_lists = zip(self.lon[0,width],nsa_lats)
            sorted_pairs = sorted(zipped_lists)
            tuples = zip(*sorted_pairs)
            list1, list2 = [ list(tuple) for tuple in  tuples]  
            plt.plot(list1, list2)
            plt.ylim([np.mean(list2)-10, np.mean(list2)+10])
            plt.show(block=False)
            plt.pause(2)
            plt.close()
        if "showShift" in self.purpose[1] and self.iter == 0:
            plt.imshow(hc_band, vmin = -20, vmax = 20)
            plt.show()
            a  = self.subset[0]
            ifPlot = hc_band[a,:]
            plt.imshow(ifPlot, vmin = -20, vmax = 20)
            plt.show()
        if "showIf" in self.purpose[1]:
            plt.imshow(self.im)
            plt.show()
            plt.imshow(hc_band)
            plt.show()
            a  = self.subset[0]
            ifPlot = hc_band[a,:]
            plt.imshow(ifPlot)
            plt.show()
        self.NSA_Analysis(nsa_lats, self.im,width)
        if "showError" in self.purpose[1] and abs(self.NSA[-1] - self.goalNSA) > self.purpose[2]:  
            if self.play == True:
                self.play = False
            else:
                for i in nsa_lats:
                    print(i)
                self.play = True
                plt.imshow(self.im)
                plt.show()
                plt.imshow(hc_band)
                plt.show()
                self.dataAnalysis()
    def ifAnalysis(self):
        #open image arrays
        try:
            self.im = plt.imread(self.currentFile)[:,:,0]
        except:
            self.im = plt.imread(self.currentFile)[:,:]
        self.im = self.im.astype(np.float32)
        self.hc_band = np.empty((self.height, self.width), float)
        if_sh = []
        for i in range(self.width):
            if_sh.append(self.if_sh_data(i))
        if_sh = np.array(if_sh).T
        #get latitude values of each pixel using CSV
        count = 0
        non_zero = np.array(np.any(self.im != 0,axis=1))
        image = if_sh[non_zero, :]
        crop = self.im[non_zero, :]
        ab = self.lat[non_zero,0]
        Result = image[:,~np.any(crop == 0, axis = 0)]
        b = Result[int(len(Result)*0.25):int(len(Result)*0.75), int(len(Result[0])*0.25):int(len(Result[0])*0.75)]
        b = np.mean(b)*10
        a = np.mean(Result, axis = 1)
        ab = ab[abs(a)<abs(b)]
        a = a[abs(a)<abs(b)]
        if self.purpose[2] == "show":
            plt.imshow(Result)
            plt.show()
            plt.title(self.flyby + " band: " + str(self.iter)) 
            plt.xlabel("latitude")
            plt.ylabel("interface")
            plt.plot(ab, a, marker = ".")
            plt.show()
        return [ab, a/255]
    def tiltAnalysis(self):
        #open image arrays
        if self.iter in self.purpose[1]:
            try:
                self.im = plt.imread(self.currentFile)[:,:,0]
            except:
                self.im = plt.imread(self.currentFile)[:,:]
            self.im = self.im.astype(np.float32)
            self.hc_band = np.empty((self.height, self.width), float)
            nsa_lats = []; nsa_lons = []; cols = []
            columns = []
            lon_shTilt = []
            for col in range(*self.purpose[2]):
                x = self.columnAnalysis(col)
                if x != None:
                    nsa_lats.append(x)
                    lon_shTilt.append(self.lon[0,col])
                    columns.append(col)
                print(x)
            combo = 5
            movingAverageList = self.running_mean(nsa_lats, combo) 
            function, angle, r_squared = self.NSATilt(lon_shTilt[combo-1::], movingAverageList)
            if "showTilt" in self.purpose[1]:
                plt.plot(lon_shTilt[combo-1::], movingAverageList)
                combo = 10
                movingAverageList = self.running_mean(nsa_lats, combo) 
                function, angle, r_squared = self.NSATilt(lon_shTilt[combo-1::], movingAverageList)
                plt.plot(lon_shTilt[combo-1::], movingAverageList)
                combo = 15
                movingAverageList = self.running_mean(nsa_lats, combo) 
                function, angle, r_squared = self.NSATilt(lon_shTilt[combo-1::], movingAverageList)
                plt.plot(lon_shTilt[combo-1::], movingAverageList)
                combo = 3
                movingAverageList = self.running_mean(nsa_lats, combo) 
                function, angle, r_squared = self.NSATilt(lon_shTilt[combo-1::], movingAverageList)
                plt.plot(lon_shTilt[combo-1::], movingAverageList)
                plt.plot(lon_shTilt, nsa_lats)
                plt.show()
            self.band.append([columns, lon_shTilt[combo-1::], movingAverageList,function, angle, r_squared])
    def linearRegress(self, x, y):
        return np.polyfit(x,y,1)
    def angle(self, slope):
        return 180/math.pi*np.arctan(slope)
    def NSATilt(self, x, y):
        function = self.linearRegress(x,y)
        x=np.array(x, dtype = "float64")
        ys =  x*function[0]+function[1]
        a = r2_score(y, ys)
        return function, self.angle(function[0]), a
    def if_sh_data(self, column):
        subtraction = (np.insert(self.im[:,column], [0]*self.num_of_nans, self.nans) - np.concatenate((self.im[:,column], self.nans)))
        self.hc_band[:,column] = subtraction[int(self.num_of_nans/2):int(-self.num_of_nans/2)]
        #hc_band[crop[0]:crop[1],crop[2]:crop[3]]
        columnHC = self.hc_band[:,column]
        if_sh = columnHC ## subset HC band b/t 30°S to 0°N 
        #lat_sh = self.columnLat[self.subset]  ## subset HC band b/t 30°S to 0°N 
        #lon_sh = self.lon[:,column][self.subset]  ## subset HC band b/t 30°S to 0°N 
        return if_sh
    def NSA_Analysis(self, im_nsa_lat,image, x):
        dev = np.std(im_nsa_lat) #standard deviation
        average = np.nanmean(im_nsa_lat) #standard average
        combo = 4
        movingAverageList = self.running_mean(im_nsa_lat, combo) #moving average
        if "showAverage" in self.purpose[1]:
            plt.plot(range(len(movingAverageList)),movingAverageList)
            plt.show()
        movingAvg = np.mean(movingAverageList) #moving average
        diff = self.brightnessDifference(image, movingAvg) #difference between north and south
        self.NSA.append(movingAvg)
        self.deviation.append(dev)
        self.NS.append(diff)
    def columnAnalysis(self,column):
        subtraction = (np.insert(self.im[:,column], [0]*self.num_of_nans, self.nans) - np.concatenate((self.im[:,column], self.nans)))
        self.hc_band[:,column] = subtraction[int(self.num_of_nans/2):int(-self.num_of_nans/2)]
        #hc_band[crop[0]:crop[1],crop[2]:crop[3]]
        columnHC = self.hc_band[:,column]
        if_sh = columnHC[self.subset] ## subset HC band b/t 30°S to 0°N 
        #lat_sh = self.columnLat[self.subset]  ## subset HC band b/t 30°S to 0°N 
        self.lon_sh = self.lon[:,column][self.subset]  ## subset HC band b/t 30°S to 0°N 
        try:
            popt, _ = curve_fit(self.poly6, self.lat_sh, if_sh)#apply sextic regression to data
            poptD = self.poly6Derivative(*popt)#get derivative of sextic regression
            derivativeRoot = np.roots(poptD) #roots (Real and imaginary) of derivative function
            derivativeRoots = derivativeRoot[np.isreal(derivativeRoot)] #remove extraneous soulutions (imaginary)
            drIndex = min(range(len(derivativeRoots)), key=lambda x: abs(derivativeRoots[x]-self.goalNSA)) #find value closest to NSA
            derivativeRoots = derivativeRoots[drIndex] #get lat of derivative root
            derivativeRoots.real #append to nsa_lats
            return derivativeRoots.real
        except:
            return None
    def analysis(self):
        #open image arrays
        try:
            self.im = plt.imread(self.currentFile)[:,:,0]
        except:
            self.im = plt.imread(self.currentFile)[:,:]
        self.im = self.im.astype(np.float32)
        self.hc_band = np.empty((self.height, self.width), float)
        nsa_lats = []; nsa_lons = []; cols = []
        #get latitude values of each pixel using CSV
        if self.purpose[0] == "data":
            for col in range(self.width):
                x = self.columnAnalysis(col)
                if x != None:
                    nsa_lats.append(x)
            self.NSA_Analysis(nsa_lats, self.im)
        elif self.purpose[0] == "tilt" and self.iter in self.purpose[1]:
            columns = []
            lon_shTilt = []
            for col in range(*self.purpose[2]):
                x = self.columnAnalysis(col)
                if x != None:
                    nsa_lats.append(x)
                    lon_shTilt.append(self.lon[0,col])
                    columns.append(col)
            function, angle, r_squared = self.NSATilt(lon_shTilt, nsa_lats)
            self.band.append([columns, lon_shTilt, nsa_lats,function, angle, r_squared])
        elif self.purpose[0] == "if_sh":
            if self.purpose[1] <= 1:
                self.purpose[1] = int(self.purpose[1] * self.width)
            else:
                column = self.purpose[1]
            while True:
                try:
                    ifs = self.if_sh_data(self.purpose[1])
                    break
                except:
                    self.purpose[1] +=1 
                    print(self.purpose[1])
            self.if_sh.append(ifs)
class ComparisonToRoman:
    def __init__(self, directory, Tdataset, shiftDegree):
        self.allDatasets = []
        self.NSA =[]
        self.NS_Flux_Ratio = []
        self.directory = directory
        self.Tdataset = Tdataset
        self.shiftDegree = shiftDegree
        self.createFigureFolder()
        self.createFileFolder()
        self.wavelengths()
        for self.i in Tdataset:
            self.allDatasets.append((self.directory[0] + + "/" + self.directory[7] + "/" + self.i[0] + self.directory[5]))
    def createFigureFolder(self):
        folderPath = self.directory[0] + "/" + self.directory[8][0]
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
        self.resultsFolder = folderPath    
    def createFileFolder(self):
        folderPath = self.directory[0] + "/" + self.directory[8][0] + "/" + self.directory[8][1]
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
        self.resultsFolder = folderPath
    def wavelengths(self):
        self.wavelength = (np.array(pd.read_csv(self.directory[0] + '/' + self.directory[1] + '/' + self.directory[4], header = None)))[0]
    def datasetDates(self):
        self.dates = []
        date = np.array(pd.read_csv(self.directory[0] + '/' + self.directory[1] + '/' + self.directory[6], header = None))
        for i in self.Tdataset:
            rowOne = date[:, 0]
            rowOne = rowOne.tolist()
            row = rowOne.index(i)
            self.dates.append(date[row,2])
    def datasetRead(self, x):
        self.data = np.array(pd.read_csv(x, header = None))
        self.NSA.append(((self.data[0])[1:-1]).astype(np.float64))
        self.NS_Flux_Ratio.append(((self.data[2])[1:-1]).astype(np.float64))
    def NSA_Figure(self,size = [12, 8], xLim = [0.3,1.1], yLim = [0.5,1.8], cMap = 'Greys', cmapMin = 0.5, cmapMax =2, xLabel = 'wavelength', yLabel = 'N/S', axisFontSize = 8, Title = "Wavelength vs. N/S flux ratio", titleFontSize = 10,legendLocation = 7,legendFontSize = 10, lineWidth = 2,dataPointStyle = ','):
        self.wavelengths()
        self.datasetDates()
        for i in self.allDatasets:
            self.datasetRead(i)
        self.datasetRead
        cMap = plt.cm.get_cmap(cMap)
        colors = np.arange(cmapMin,(cmapMax+((cmapMax-cmapMin)/len(self.Tdataset))),(cmapMax-cmapMin)/len(self.Tdataset))
        if cmapMax > 1:
            cmapMax = 1
        plt.rcParams["font.family"] = 'monospace'
        plt.rcParams["font.weight"] = 'light'
        for i in range(len(self.Tdataset)):
            plt.figure(num=1, figsize=size, dpi=80, facecolor='w', edgecolor='k')
            plt.xlim(xLim)
            plt.ylim(yLim)
            plt.xticks(fontsize = axisFontSize/1.2)
            plt.yticks(fontsize = axisFontSize/1.2)
            plt.xlabel(xLabel, fontsize = axisFontSize);plt.ylabel(yLabel, fontsize = axisFontSize)
            plt.title(Title, fontsize = titleFontSize*1.25)
            plt.plot(self.wavelength, self.NSA[i], label = (self.Tdataset[i] + ' - ' + self.dates[i]), color = cMap(colors[i]), linewidth=lineWidth, marker = dataPointStyle)
            plt.legend(fontsize = legendFontSize, bbox_to_anchor=(1, 0.22),  ncol = int(len(self.Tdataset)/5), markerscale = 2, framealpha = 0, frameon = False, edgecolor = 'white')
            plt.show()
    def tiltPlot(self,tiltDatasets, size = [12, 8], columns = [250, 600], xLabel = 'Longitude (°)', yLabel = 'North South Boundary Latitude (°)', axisFontSize = 8, Title = "Wavelength vs. N/S flux ratio", titleFontSize = 10, lineWidth = 2, dataPointStyle = ','):
            self.wavelengths()
            for i in range(len(tiltDatasets)):
                a = data(*['C:/Users/aadvi/Desktop/Titan Paper/Data', 'Titan Data', 'csv/', 'vis.cyl/', 4, -11], tiltDatasets[i], ["tilt", [74], columns])        
                xPlot = list(a.band[0][1])
                yPlot = list(a.band[0][2])
                yLine = [x*a.band[0][3][0]+a.band[0][3][1] for x in xPlot]
                plt.rcParams["font.family"] = 'monospace'
                plt.rcParams["font.weight"] = 'light'
                plt.figure(num = 1, figsize = size, dpi = 80, facecolor='w', edgecolor = 'k')
                plt.xlabel(xLabel, fontsize = axisFontSize);plt.ylabel(yLabel, fontsize = axisFontSize)
                if len(tiltDatasets) > 1:
                    title = Title + tiltDatasets[i] + " dataset at wavelength " + str(self.wavelength[74]) + 'µm'
                    plt.title(title, fontsize = titleFontSize*1.25)
                else:
                    plt.title(Title, fontsize = titleFontSize*1.25)
                plt.plot(xPlot, yPlot, linewidth=lineWidth, marker = dataPointStyle)
                plt.plot(xPlot, yLine, linewidth=lineWidth, marker = dataPointStyle)
                plt.figtext(0.4,0.12,"The angle of the data is "+ str(a.band[0][4]))
                plt.show()
class if_sh_Figure:
    def __init__(self, directory, Tdataset, shiftDegree):
        self.allDatasets = []
        self.NSA =[]
        self.NS_Flux_Ratio = []
        self.directory = directory
        self.Tdataset = Tdataset
        self.shiftDegree = shiftDegree
        self.wavelengths()
        self.createFigureFolder()
        self.createFileFolder()
    def createFigureFolder(self):
        folderPath = self.directory[0] + "/" + self.directory[8][0]
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
        self.resultsFolder = folderPath    
    def createFileFolder(self):
        folderPath = self.directory[0] + "/" + self.directory[8][0] + "/" + self.directory[8][2]
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
        self.resultsFolder = folderPath
    def wavelengths(self):
        self.wavelength = (np.array(pd.read_csv(self.directory[0] + '/' + self.directory[1] + '/' + self.directory[4], header = None)))[0]
    def datasetDates(self):
        self.dates = []
        date = np.array(pd.read_csv(self.directory[0] + '/' + self.directory[1] + '/' + self.directory[6], header = None))
        for i in self.Tdataset:
            rowOne = date[:, 0]
            rowOne = rowOne.tolist()
            row = rowOne.index(i[0])
            self.dates.append(date[row,2])
    def aIF(self, title, xLabel, yLabel, bands, size, cMap, axisFontSize, titleFontSize,legendLocation, legendFontSize, lineWidth, dataPointStyle, lineStyles, grid, cmapMin = 0, cmapMax = 1):
        self.wavelengths()
        self.datasetDates()
        purpose = ["if_sh",bands, ""]
        plt.rcParams["font.family"] = 'monospace'
        plt.rcParams["font.weight"] = 'light'
        xTicks = range(-50, 51, 10)
        yTicks = np.arange(-0.1, 0.125, 0.025)
        cMap = plt.cm.get_cmap(cMap)
        colors = np.arange(cmapMin,(cmapMax+((cmapMax-cmapMin)/len(bands))),(cmapMax-cmapMin)/len(bands))
        fig, axs = plt.subplots(nrows = math.ceil(len(self.Tdataset)/grid), ncols = grid, sharex='all', sharey='all', squeeze = False, figsize = size)
        plt.subplots_adjust(hspace = 0.4)
        axs = axs.ravel()
        for i in range(len(self.Tdataset)):
            currentDataset = data(self.directory,self.Tdataset[i], self.shiftDegree, purpose)
            for band in range(len(bands)):
                x = currentDataset.if_sh[band][0]
                y = currentDataset.if_sh[band][1]
                if i == 0:
                    axs[i].plot(x,y, lw = lineWidth, linestyle = lineStyles[band], color = cMap(colors[band]), label = str(self.wavelength[band]) +"µm" )
                else:
                    axs[i].plot(x,y, lw = lineWidth, linestyle = lineStyles[band], color = cMap(colors[band]))
            axs[i].set_ylabel(yLabel)
            setTick = max(abs(x))
            axs[i].set_xticks(xTicks)
            axs[i].set_yticks(yTicks)
            axs[i].set_xlim(min(xTicks),max(xTicks))
            axs[i].set_ylim(min(yTicks),max(yTicks))
            axs[i].minorticks_on()
            axs[i].xaxis.set_tick_params(labelbottom=True)
            axs[i].yaxis.set_tick_params(labelleft=True)
            axs[i].text(0.94, 0.8, self.Tdataset[i][0], horizontalalignment='center', verticalalignment='center', transform=axs[i].transAxes)
            if i + grid >= len(self.Tdataset):
                axs[i].set_xlabel(xLabel)
        remainder = len(self.Tdataset) % grid
        for x in range(i+1, i+1+remainder):
            fig.delaxes(axs[x])
        fig.suptitle(title)
        lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        fig.legend(lines, labels, loc = legendLocation)
        plt.show()
    def bIF(self, title, xLabel, yLabel, bands, size, cMap, axisFontSize, titleFontSize,legendLocation, legendFontSize, lineWidth, dataPointStyle,lineStyles, grid, cmapMin = 0, cmapMax = 1):
        self.wavelengths()
        self.datasetDates()
        purpose = ["if_sh",bands, ""]
        plt.rcParams["font.family"] = 'monospace'
        plt.rcParams["font.weight"] = 'light'
        grid = 2
        xTicks = range(-90, 91, 10)
        yTicks = np.arange(-30, 31, 15)
        cMap = plt.cm.get_cmap(cMap)
        colors = np.arange(cmapMin,(cmapMax+((cmapMax-cmapMin)/len(self.Tdataset))),(cmapMax-cmapMin)/len(bands))
        plt.figure(size)
        fig, axs = plt.subplots(nrows = math.ceil(len(self.Tdataset)/grid), ncols = grid, sharex='all', sharey='all', squeeze = False, figsize = size)
        plt.subplots_adjust(hspace = 0.4)
        axs = axs.ravel()
        for i in range(len(self.Tdataset)):
            currentDataset = data(self.directory,self.Tdataset[i], self.shiftDegree, purpose)
            for band in range(len(bands)):
                x = currentDataset.if_sh[band][0]
                y = currentDataset.if_sh[band][1]
                if i == 0:
                    axs[i].plot(x,y, lw = lineWidth, linestyle = lineStyles[band], color = cMap(colors[band]), label = str(self.wavelength[band]) +"µm" )
                else:
                    axs[i].plot(x,y, lw = lineWidth, linestyle = lineStyles[band], color = cMap(colors[band]))
            axs[i].set_ylabel(yLabel)
            setTick = max(abs(x))
            axs[i].set_xticks(xTicks)
            axs[i].set_yticks(yTicks)
            axs[i].set_xlim(min(xTicks),max(xTicks))
            axs[i].set_ylim(min(yTicks),max(yTicks))
            axs[i].minorticks_on()
            axs[i].xaxis.set_tick_params(labelbottom=True)
            axs[i].yaxis.set_tick_params(labelleft=True)
            axs[i].title.set_text(self.Tdataset[i][0])
            if i + grid >= len(self.Tdataset):
                axs[i].set_xlabel(xLabel)
        remainder = len(self.Tdataset) % grid
        for x in range(i+1, i+1+remainder):
            fig.delaxes(axs[x])
        fig.suptitle(title)
        lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        fig.legend(lines, labels, loc = legendLocation)
        plt.show()
class NS_Flux_Ratio:
    def __init__(self, directory, Tdataset, shiftDegree):
        self.allDatasets = []
        self.NSA =[]
        self.NS_Flux_Ratio = []
        self.directory = directory
        self.Tdataset = Tdataset
        self.shiftDegree = shiftDegree
        self.wavelengths()
        self.createFigureFolder()
        self.createFileFolder()
        for self.i in Tdataset:
            self.allDatasets.append((self.directory[0] + "/" + self.directory[7] + "/" + self.i[0] + self.directory[5]))
    def createFigureFolder(self):
        folderPath = self.directory[0] + "/" + self.directory[8][0]
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
        self.resultsFolder = folderPath    
    def createFileFolder(self):
        folderPath = self.directory[0] + "/" + self.directory[8][0] + "/" + self.directory[8][3]
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
        self.resultsFolder = folderPath
    def wavelengths(self):
        self.wavelength = (np.array(pd.read_csv(self.directory[0] + '/' + self.directory[1] + '/' + self.directory[4], header = None)))[0]
    def datasetDates(self):
        self.dates = []
        date = np.array(pd.read_csv(self.directory[0] + '/' + self.directory[1] + '/' + self.directory[6], header = None))
        for i in self.Tdataset:
            rowOne = date[:, 0]
            rowOne = rowOne.tolist()
            row = rowOne.index(i[0])
            self.dates.append(date[row,2])
    def datasetRead(self, x):
        self.data = np.array(pd.read_csv(x, header = None))
        self.NSA.append(((self.data[0])[1:-1]).astype(np.float64))
        self.NS_Flux_Ratio.append(((self.data[2])[1:-1]).astype(np.float64))
    def aNS_Flux(self, title, xLabel, yLabel, size, xLim, yLim, cMap, cmapMin, cmapMax, axisFontSize, titleFontSize,legendLocation,legendFontSize, lineWidth,dataPointStyle, lineStyles, grid, subplotName):
        self.datasetDates()
        for i in self.allDatasets:
            self.datasetRead(i)
        plt.rcParams["font.family"] = 'times'
        plt.rcParams["font.weight"] = 'light'
        xtickCount = 10
        yTickCount =4
        xTicks = np.arange(xLim[0], xLim[1]+(xLim[1]-xLim[0])/xtickCount,(xLim[1]-xLim[0])/xtickCount)
        yTicks = np.arange(yLim[0], yLim[1]+(yLim[1]-yLim[0])/yTickCount,(yLim[1]-yLim[0])/yTickCount)
        fig, axs = plt.subplots(nrows = math.ceil(len(self.Tdataset)/grid), ncols = grid, sharex='all', sharey='all', squeeze = False, figsize = size)
        cMap = plt.cm.get_cmap(cMap)
        try:
            colors = np.arange(cmapMin,(cmapMax+((cmapMax-cmapMin)/len(self.Tdataset))),(cmapMax-cmapMin)/len(self.Tdataset))
        except: 
            colors = [cmapMax for i in range(len(self.Tdataset))]
        axs = axs.ravel()
        for i in range(len(self.Tdataset)):
            if i > 0:
                plt.subplots_adjust(hspace = 0.4)
            axs[i].plot(self.wavelength, self.NS_Flux_Ratio[i], color = cMap(colors[i]), linewidth=lineWidth, marker = dataPointStyle)
            axs[i].plot([-1000,1000], [1,1], color = 'red', linewidth = lineWidth/4, linestyle = 'dashed')
            axs[i].set_ylabel(yLabel)
            axs[i].set_yticks(yTicks)
            axs[i].set_xlim(xLim)
            axs[i].set_ylim(yLim)
            axs[i].minorticks_on()
            if i == len(self.Tdataset)-1:
                axs[i].set_xticks(xTicks)
                axs[i].xaxis.set_tick_params(labelbottom=True)
            axs[i].yaxis.set_tick_params(labelleft=True)
            axs[i].text(*subplotName, (self.Tdataset[i][0] + ' - ' + self.dates[i]), horizontalalignment='center', verticalalignment='center', transform=axs[i].transAxes)
            if i + grid >= len(self.Tdataset):
                axs[i].set_xlabel(xLabel)
        remainder = len(self.Tdataset) % grid
        for x in range(i+1, i+1+remainder):
            fig.delaxes(axs[x])
        fig.suptitle(title)
        #lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        #lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        #fig.legend(fontsize = legendFontSize, loc = legendLocation, markerscale = 2, framealpha = 0, frameon = False, edgecolor = 'white', handlelength = 0)
        plt.show()
class Tilt: 
    def __init__(self, directory, Tdataset, dataInfo):
        self.allDatasets = []
        self.NSA =[]
        self.NS_Flux_Ratio = []
        self.directory = directory
        self.Tdataset = Tdataset
        self.shiftDegree = dataInfo
        self.createFigureFolder()
        self.createFileFolder()
        self.wavelengths()
        self.createFigureFolder()
        self.createFileFolder()
        for self.i in Tdataset:
            self.allDatasets.append((self.directory[0] + "/" + self.directory[7] + "/" + self.i[0] + self.directory[5]))
    def createFigureFolder(self):
        folderPath = self.directory[0] + "/" + self.directory[8][0]
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
        self.resultsFolder = folderPath    
    def createFileFolder(self):
        folderPath = self.directory[0] + "/" + self.directory[8][0] + "/" + self.directory[8][4]
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
        self.resultsFolder = folderPath
    def wavelengths(self):
        self.wavelength = (np.array(pd.read_csv(self.directory[0] + '/' + self.directory[1] + '/' + self.directory[4], header = None)))[0]
    def datasetDates(self):
        self.dates = []
        date = np.array(pd.read_csv(self.directory[0] + '/' + self.directory[1] + '/' + self.directory[6], header = None))
        for i in self.Tdataset:
            rowOne = date[:, 0]
            rowOne = rowOne.tolist()
            row = rowOne.index(i)
            self.dates.append(date[row,2])
    def datasetRead(self, x):
        self.data = np.array(pd.read_csv(x, header = None))
        self.NSA.append(((self.data[0])[1:-1]).astype(np.float64))
        self.NS_Flux_Ratio.append(((self.data[2])[1:-1]).astype(np.float64))
    def atiltPlot(self, tiltDatasets, bands, size, columns, xLabel , yLabel, axisFontSize, Title, titleFontSize, lineWidth, dataPointStyle, angle):
            self.wavelengths()
            purpose = ["tilt",bands,columns]
            for i in range(len(tiltDatasets)):
                dataTdataset = self.Tdataset[tiltDatasets[i]]
                a = data(self.directory, dataTdataset, self.shiftDegree, purpose)  
                for bander in range(len(bands)):      
                    xPlot = list(a.band[bander][1])
                    yPlot = list(a.band[bander][2])
                    yLine = [x*a.band[bander][3][0]+a.band[bander][3][1] for x in xPlot]
                    plt.rcParams["font.family"] = 'monospace'
                    plt.rcParams["font.weight"] = 'light'
                    plt.figure(num = 1, figsize = size, dpi = 80, facecolor='w', edgecolor = 'k')
                    plt.xlabel(xLabel, fontsize = axisFontSize);plt.ylabel(yLabel, fontsize = axisFontSize)
                    if len(tiltDatasets) > 1:
                        title = Title + self.Tdataset[tiltDatasets[i]][0] + " dataset at wavelength " + str(self.wavelength[74]) + 'µm'
                        plt.title(title, fontsize = titleFontSize*1.25)
                    else:
                        plt.title(Title, fontsize = titleFontSize*1.25)
                    plt.plot(xPlot, yPlot, linewidth=lineWidth, marker = dataPointStyle)
                    plt.plot(xPlot, yLine, linewidth=lineWidth, marker = dataPointStyle)
                    plt.figtext(0.4,0.12, angle+ str(a.band[0][4]) + "°")
                    plt.show()
class printDatasets:
    def __init__(self, directory,Tdatasets,print):
        if print[1] == "all":
            self.whichFlyby = True
        else:
            self.whichFlyby = print[1]
        if print[2] == "all":
            self.row = True
        else:
            self.row = print[2]
        self.directory = directory
        self.Tdatasets = Tdatasets
        self.file = []
        self.mainPrint()
    def mainPrint(self):
        for i in self.Tdatasets:
            if self.whichFlyby == True or i in self.whichFlyby:
                file = (self.directory[0] + "/" + self.directory[7] + "/" + i + self.directory[5])
                self.print(file)
                input("input anything for next dataset")
    def print(self, file):
        print(file, "\n\n\n\n")
        self.file.append(np.array(pd.read_csv(file, header = None)))
        for i in range(len(self.file[-1])):
            if self.row == True or i in self.row:
                for x in self.file[-1][i]:
                    print(x)
                input("input anything for next filetype")
class Titan:
    def __init__(self, directory = ['C:/Users/aadvi/Desktop/Titan Paper/Data', 'Titan Data', 'csv', 'vis.cyl', 'wavelength.csv','_analytics.csv', 'nsa_cubes_northernsummer.csv', 'Result', ['Figures', 'ComparisonToRoman', 'IF Subplots', 'NS Flux Ratio', 'Tilt']], shiftDegree = 6, datasets =  [['Ta', -12, 15],['T8', -12, 15],['T31', -12, 15],['T61', -12, 15],['T62', -12, 15],['T67', -12, 15],['T79', -10, 15],['T85', -10, 15],['T92', -5, 15],['T108', -0, 15],['T114', 0, 20]], purpose = ["if_sh", [71,72,73], "show"], whichDatasets = True,  info = []):
        self.directory = directory
        self.datasets = list(np.array(datasets)[:,0])
        self.datasetsNSA = datasets
        self.allDatasets = [] 
        self.allFiles = []
        self.purpose = purpose
        self.shiftDegree = shiftDegree
        self.information = info
        if whichDatasets == "all":
            self.which = True
        else:
            self.which = whichDatasets
        if self.purpose[0] == "data" or self.purpose[0] == "tilt" or self.purpose[0] == "if_sh":
            self.getData()
        elif self.purpose[0] == "figure" and self.purpose[1] == "comparison":
            self.fig4()
        elif self.purpose[0] == "figure" and self.purpose[1] == "if":
            self.fig5()
        elif self.purpose[0] == "figure" and self.purpose[1] == "flux":
            self.fig6()
        elif self.purpose[0] == "figure" and self.purpose[1] == "tilt":
            self.fig8()
        elif self.purpose[0] == "printcsv":
            self.printCSV()
    def getData(self):
        for i in range(len(self.datasets)):
            if self.which == True or self.datasets[i] in self.which:
                x = data(self.directory,self.datasetsNSA[i], self.shiftDegree, self.purpose) 
                print(i)
    def fig4(self):
        pass
    def fig5(self):
        x = if_sh_Figure(self.directory, self.datasetsNSA, self.shiftDegree)
        if self.information == 0:
            x.aIF(title = "Seasonal Evolution of Titan's Atmospheric Meridional Brightness Profile", xLabel = "Latitude", yLabel = "I/F", bands = [24,35,50], size = [16,16], cMap = "viridis", axisFontSize = 10, titleFontSize = 15, legendLocation = 4, legendFontSize = 6, lineWidth = 1, dataPointStyle = ".", lineStyles = ["solid", "solid", "solid"], grid = 1, cmapMin = 0, cmapMax = 1)
        elif self.information == 1:
            pass
        elif self.information == 2:
            pass
        elif self.information == 3:
            pass
    def fig6(self):
        x = NS_Flux_Ratio(self.directory, self.datasetsNSA, self.shiftDegree)
        if self.information == 0:
            x.aNS_Flux(title = "Seasonal Evolution of Titan's Atmospheric Meridional Brightness Profile", xLabel = "Wavelength (µm)", yLabel = "N/S", size = [24,24], xLim = [0.35,1.04], yLim = [0.6,2], cMap = "binary", cmapMin = 0.9, cmapMax = 0.9, axisFontSize = 10, titleFontSize = 10,legendLocation = 4, legendFontSize = 9, lineWidth = 1.5, dataPointStyle = ",", lineStyles = ["solid", "solid", "solid"], grid = 1, subplotName = [0.06,0.9])
        elif self.information == 1:
            x.aNS_Flux(title = "Seasonal Evolution of Titan's Atmospheric Meridional Brightness Profile", xLabel = "Wavelength (µm)", yLabel = "N/S", size = [12,12], xLim = [0.35,1.04], yLim = [0.6,2], cMap = "binary", cmapMin = 0.9, cmapMax = 0.9, axisFontSize = 10, titleFontSize = 10,legendLocation = 4, legendFontSize = 9, lineWidth = 1.5, dataPointStyle = ",", lineStyles = ["solid", "solid", "solid"], grid = 1, subplotName = [0.06,0.83])   
        
        elif self.information == 2:
            pass
        elif self.information == 3:
            pass   
    def fig8(self):
        x = Tilt(self.directory, self.datasetsNSA, self.shiftDegree)
        if self.information == 0:
            x.atiltPlot(tiltDatasets = [3, 5], bands = [74, 89], size = [12, 8], columns = [400, 600], xLabel = 'Longitude (°)', yLabel = 'North South Boundary Latitude (°)', axisFontSize = 12, Title = "Axis Tilt of NSA found in the ", titleFontSize = 10, lineWidth = 2, dataPointStyle = ',', angle = "The NSA is ")
        elif self.information == 1:
            x.atiltPlot(tiltDatasets = [3, 5], bands = [74, 89], size = [12, 8], columns = [400, 600], xLabel = 'Longitude (°)', yLabel = 'North South Boundary Latitude (°)', axisFontSize = 12, Title = "Axis Tilt of NSA found in the ", titleFontSize = 10, lineWidth = 2, dataPointStyle = ',', angle = "The NSA is ")
        elif self.information == 2:
            pass
        elif self.information == 3:
            pass
        
    def printCSV(self):
        printDatasets(self.directory, self.datasets, self.purpose)
#input purpose here. List must be length 3.
#Titan(purpose = ["figure", "comparison"], info = 0, whichDatasets = "all")
#Titan(purpose = ["figure", "if"], info = 0, whichDatasets = "all")
#Titan(purpose = ["figure", "flux"], info = 0, whichDatasets = "all")
#Titan(purpose = ["figure", "tilt"], info = 0, whichDatasets = "all")]
Titan(purpose = ["figure", "flux"], info = 1, whichDatasets = "all")
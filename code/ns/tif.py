import tkinter as tk
import tkinter.filedialog as fd
from PIL import Image, ImageStat
import os
import os.path as path
import numpy as np
import pandas as pd
import csv
class tif:
    def __init__(self):
        self.dataset = input("input dataset name")
        self.nS = int(input("input width"))
        self.nL = int(input("input height"))
        root = tk.Tk()
        self.file = fd.askdirectory(parent=root, title='Choose root data folder') + '/'
        self.visCylLocation = fd.askopenfilename(parent=root, title='Choose vis.cyl csv location')
        self.csvLocation = fd.askopenfilename(parent=root, title='Choose csv location')
        self.createDatasetFolder()
        self.createCSVFolder()
        self.createVisCylFolder()
        root.quit()
        self.resultsFolder = self.datasetFolder + '/vis.cyl/'
        try:
            x = open("C:\\Users\\aadvi\\Desktop\\Titan Paper\\Data\\Titan Data\\IFDatas.csv","a+", newline='')
        except:
            x = open("C:\\Users\\aadvi\\Desktop\\Titan Paper\\Data\\Titan Data\\IFDatas.csv","w", newline='')
        self.analysisFile = x
        a = csv.writer(self.analysisFile)  
        self.dataMultiply = []
        for currentBand in range(96):
            band = self.Jcube(csv = self.visCylLocation, nL = self.nL, nS = self.nS, bands = currentBand)
            self.dataMultiply.append((255/np.max(band)))
            im = Image.fromarray(np.uint8(band * (255/np.max(band))) , 'L')
            #im = Image.fromarray(np.uint8(band) , 'L')
            if currentBand < 10:    
                saveBand = "0" + str(currentBand)
                im.save((self.resultsFolder + str(self.dataset) + "." + saveBand + '.tif'))
            else:
                im.save((self.resultsFolder + str(self.dataset) + "." + str(currentBand) + '.tif'))
        a.writerow(self.dataMultiply)
        self.analysisFile.close()
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
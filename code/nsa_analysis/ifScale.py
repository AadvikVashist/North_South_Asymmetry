import tkinter as tk
import tkinter.filedialog as fd
from PIL import Image, ImageStat
import os
import os.path as path
import numpy as np
import pandas as pd
import csv
class ifScale:
    def __init__(self, tdatass):
        root = tk.Tk()
        self.Tdatasets = tdatass
        self.Tdata = []
        for i in self.Tdatasets:
            self.Tdata.append(fd.askopenfilename(parent=root, title='Choose location' + i[0]))
        root.quit()
        open("C:\\Users\\aadvi\\Desktop\\Titan Paper\\Data\\Titan Data\\IFDatas.csv","w")
        x = open("C:\\Users\\aadvi\\Desktop\\Titan Paper\\Data\\Titan Data\\IFDatas.csv","a+", newline='')
        self.analysisFile = x
        a = csv.writer(self.analysisFile)
        for data in range(len(self.Tdatasets)):
            self.dataset = self.Tdatasets[data][0]
            self.nS = self.Tdatasets[data][1][0]
            self.nL = self.Tdatasets[data][1][1]
            dataMultiply = []
            for currentBand in range(96):
                band = self.Jcube(csv = self.Tdata[data], nL = self.nL, nS = self.nS, bands = currentBand)
                #im = Image.fromarray(np.uint8(band * (255/np.max(band))) , 'L')
                dataMultiply.append(255/np.max(band))
            a.writerow(dataMultiply)
        self.analysisFile.close()

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
x = ifScale([["T85",[725,361]],["62",[725,361]]])
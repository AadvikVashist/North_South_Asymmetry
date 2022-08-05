## import packages and set up pyvims from google drive

import pandas as pd; import numpy as np; import itertools; import matplotlib.pyplot as plt; import re
#import pyvims;
import pyvims
import cv2; from pyvims.projections import equi_cube
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
def destriping(d, band):
    ## Example code - T85 cube 
    cub = pyvims.VIMS('1721919127_1', channel='vis')   
    cub.data[band] = Jcube_to_csv(csv = d + 't85_test.csv', bands=band, nL=cub.NL, nS=cub.NS)
    ## start of analysis (datasets)
    cubData =cub.data[band]
    cubDataBand = cub.data[band][:,:]
    #column = cub.data[band][:,:]
    surface = cub.data[band][:,:][~cub.limb[:,:]]
    mask = cub.data[band]
    limb = cub.data[band][(cub.alt < 300.0) & (cub.alt > 0.1)]
    kernel_size_x = 1;kernel_size_y = 3
    lowpass_kernel_box = np.ones((kernel_size_x, kernel_size_y))
    lowpass_kernel_box = lowpass_kernel_box / (kernel_size_x * kernel_size_y)
    tt=73
    cub.data[tt] = cv2.filter2D(cub.data[tt], -1, lowpass_kernel_box)
    cyl, (x,y), extent, cnt = equi_cube(cub, tt+1, n=512, interp='cubic')
    plt.imshow(cyl)
    plt.show()
    """plt.imshow(cubData)
    plt.show()
    plt.imshow(cub.alt)
    plt.show()
    plt.imshow(cub.limb)
    plt.show()"""
    plt.imshow(mask, vmin= 0, vmax = 0.1)
    plt.colorbar()
    plt.show() 
    #remove extraneous mask pixels
    limbs = np.where(cub.alt > 10, True, False) 
    mask = np.where(limbs, np.nan, cubData)
    median = np.nanmedian(mask) #median = np.median(cubData) 
    x , y = mask.shape
    import time
    plt.imshow(mask, vmin= 0, vmax = 0.1)
    plt.colorbar()
    plt.show() 
destriping('C:/Users/aadvi/Desktop/Titan Paper/Data/Raw T85/', 74)
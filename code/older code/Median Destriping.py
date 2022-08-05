## import packages and set up pyvims from google drive

import pandas as pd; import numpy as np; import itertools; import matplotlib.pyplot as plt; import re
#import pyvims;
import pyvims

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
def destriping(d, rangeX, rangeY):
    ## Example code - T85 cube 
    cub = pyvims.VIMS('1721919127_1', channel='vis')   
    realMask = cub.data[1] = Jcube_to_csv(csv = d + 't85_test.csv',        ## replace pyvims image w/ Jcube csv data 
                                    bands=1, nL=cub.NL, nS=cub.NS)
    realMask = np.empty(realMask.shape) #full mask
    realUnchangedMask = np.empty(realMask.shape)
    for band in range(rangeX, rangeY+1): # wavelength index

                   ## import pyvims cube data (only visible bands 1-96)
        cub.data[band] = Jcube_to_csv(csv = d + 't85_test.csv',        ## replace pyvims image w/ Jcube csv data 
                                    bands=band, nL=cub.NL, nS=cub.NS)

        ## start of analysis (datasets)
        col = 30
        cubData =cub.data[band]
        cubDataBand = cub.data[band][:,col]
        #column = cub.data[band][:,col]
        surface = cub.data[band][:,col][~cub.limb[:,col]]
        mask = cub.data[band][:,col][(cub.alt[:,col] > 300.0) & (cub.limb[:,col])]
        limb = cub.data[band][:,col][(cub.alt[:,col] < 300.0) & (cub.alt[:,col] > 0.1)]
        """plt.imshow(cubData)
        plt.show()
        plt.imshow(cub.alt)
        plt.show()
        plt.imshow(cub.limb)
        plt.show()"""

        #remove extraneous mask pixels
        limbs = np.where(cub.alt > 10, True, False) 
        mask = np.where(limbs, np.nan, cubData)
        median = np.median(mask[~np.isnan(mask)]) #median = np.median(cubData) 
        x , y = mask.shape
        import time
        newMask = np.empty((x,y)) 
        for a in range(x): #iterates over columns
            column = mask[a, :] 
            maskPixels = column[~np.isnan(column)]
            columnMedian = np.median(maskPixels)
            fixValue = median - columnMedian
            if np.isnan(fixValue):
                fixValue = 0
            print(fixValue)
            for b in range(y):
                pixel = column[b]
                if np.invert(np.isnan(pixel)):
                    pixelChangeValue = pixel + fixValue
                    newMask[a,b] = pixelChangeValue
                else:
                    pixelChangeValue = 0
                    newMask[a,b] = np.nan

            
        columnValue = 10
        """"
        fixValue = z(columnValue)
        plt.imshow(mask, vmin= 0, vmax = 0.1)
        plt.colorbar()
        plt.show()
        plt.savefig('x.png')
        plt.imshow(newMask, vmin= 0, vmax = 0.1)
        plt.colorbar()
        plt.savefig('y.png')
        plt.show()
        """

        for i in range(x):
            for j in range(y):
                realMask[i, j] += newMask[i,j]
                realUnchangedMask[i, j] += mask[i, j]
        ## visualize different datasets
        """plt.figure(figsize=(10,10))
        plt.imshow(np.where(cub.alt>0.1,np.where(cub.alt<300.0,cub[band],np.nan),np.nan),cmap="Blues")
        plt.colorbar()
        plt.contour(cub.alt, [1],colors='g', linewidths=2, linestyles='--') ## surface contour
        alts = plt.contour(cub.alt, [300], colors=['r','orange'])
        fmt = {}; strs = ['300 km', '400 km']
        for l, s in zip(alts.levels, strs):
            fmt[l] = s
            #plt.clabel(alts, alts.levels, use_clabeltext=True, fmt=fmt)
            titanPlot = np.where(cub.alt>300.,cub[band],np.nan)
            plt.imshow(titanPlot,vmax=0.005);plt.colorbar();plt.title('limb, 0-300 km')
            plt.imshow(np.where(~cub.limb,cub[band],np.nan),cmap='Greys_r');plt.colorbar()
            ## easier to see striping
            plt.figure(); cub.plot(band+1, 'equi',vmin=0.03)"""
    plt.imshow(realMask, vmin = 0, vmax = 10)
    plt.colorbar()
    plt.show()
    plt.imshow(realUnchangedMask, vmin = 0, vmax = 10)
    plt.colorbar()
    plt.show()
    return realUnchangedMask, realMask
frame, frameEditted = destriping('C:/Users/aadvi/Desktop/Titan Paper/Data/Raw T85/', 1, 94)
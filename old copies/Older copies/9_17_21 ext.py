#Wait commands
from re import X
import time
#iterate through directory
import os, os.path
#image analysis
from PIL import Image, ImageStat
#Regressions and Plotting
import matlab
import pylab
import numpy as np; import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.mlab as mlab
from matplotlib.ticker import PercentFormatter
import scipy
from scipy import stats
from scipy.stats import mode
from scipy.stats import norm
from scipy.optimize import curve_fit
import statistics
#other
import warnings 
import pyvims
import pygsheets
import sys
#fix runtime errors
def fxn():
    warnings.warn("RunTime", RuntimeWarning)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
#jcube to csv
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
def brightness( im_file ):
    im = Image.open(im_file).convert('L')
    stat = ImageStat.Stat(im)
    return stat.mean[0]
def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / 4 / stddev)**2)
def poly6(x, g, h, i, j, a, b, c):
    return g*x**6+h*x**5+i*x**4+j*x**3+a*x**2+b*x+c
def shift (csv, image, lat, lon, crop):
    ## read cyl image 
    im = plt.imread(image)
    im = im.astype(np.float32)
    ## create high-contrasting band image (I/F difference columns)
    hc_band = np.empty((125,133), float)
    nsa_lats = []; nsa_lons = []
    cols = []
    for col in range(133):
        ## construct column of HC band image
        shift_IF_down  = np.insert(im[:,col], [0,0], [np.nan,np.nan])
        shift_IF_up  = np.concatenate((im[:,col],[np.nan],[np.nan]))
        hc_band[:,col] = (shift_IF_up - shift_IF_down)[1:-1]
        ## subset HC band column b/t 25°S to 0°N 
        hc_band[crop[0]:crop[1],crop[2]:crop[3]]
        if_sh = (shift_IF_up-shift_IF_down)[(np.concatenate(([np.nan],lat[:,col],[np.nan])) > -25.0) &\
                                    (np.concatenate(([np.nan],lat[:,col],[np.nan])) < 0.0)]
        lat_sh = (np.concatenate(([np.nan],lat[:,col],[np.nan])))[(np.concatenate(([np.nan],lat[:,col],[np.nan])) > -25.0) &\
                                    (np.concatenate(([np.nan],lat[:,col],[np.nan])) < 0.0)]
        lon_sh = (np.concatenate(([np.nan],lon[:,col],[np.nan])))[(np.concatenate(([np.nan],lat[:,col],[np.nan])) > -25.0) &\
                                    (np.concatenate(([np.nan],lat[:,col],[np.nan])) < 0.0)]
        
        ## plot of model fit with column array from image
        ## apply 6th-order polyfit to I/F brightness profile
        popt, _ = curve_fit(poly6, lat_sh, if_sh)
        ## find and record latitude of minimum I/F in 6th-order polyfit/
        try:
            if (col < crop[3]) & (crop[2] <= col):
                nsa_lats.append(lat_sh[poly6(lat_sh, *popt) == np.max(np.abs(poly6(lat_sh, *popt)))][0])
                nsa_lons.append(lon_sh[poly6(lat_sh, *popt) == np.max(np.abs(poly6(lat_sh, *popt)))][0])
                cols.append(col)
        except:
            pass
    #lats = csv lat
    lats = lat.tolist()
    latlist = []
    x=0
    plottedlat = []
    for i in lats:
        latlist.append(lat[x,0])
        x+=1
    for i in range(len(nsa_lats)):
        for x in range(len(latlist)):
            if nsa_lats[i] == latlist[x]:
                plottedlat.append(x)

    slope, intercept, r, p, se = stats.linregress(cols, plottedlat)
    #line
    f = lambda x:  intercept + slope*x-crop[0]
    z = lambda x:  x - crop[ 0]
    yaxis = [f(x) for x in cols]
    points = [z(x) for x in plottedlat]
    #histogram
    avg = np.mean(points)
    var = np.var(points)
    # shape of the fitted Gaussian.
    if len(points) < 1 or var <= 1: 
        print("not enough computable values for gaussian")
        return(0, 0, 0, 0, False)
    pdf_x = np.linspace(np.min(points),np.max(points),100)
    pdf_y = 1.0/np.sqrt(2*np.pi*var)*np.exp(-0.5*(pdf_x-avg)**2/var)

    #histogram plotting :
    plt.figure()
    plt.hist(points,30,density=True)
    plt.plot(pdf_x,pdf_y,'k--')
    """plt.legend(("Fit","Data"),"best")
    plt.show()  """
    plt.close('all') 
    #lat conversions
    Histmax = np.argmax(pdf_y) 
    preLat = int(pdf_x[Histmax] + crop[0]) #crop[0] comes from lambda z
    histOut = lat[preLat][0]

    return nsa_lats, nsa_lons, hc_band, histOut, True
def ImageToAnalyze (image, directoryList, cropParam, csv, lat, lon):
    #print(image)
    nsa_lats, nsa_lons, uncroppedimgdata, histMaxLat, computable  = shift(csv, image,lat, lon,cropParam)
    if computable == True:
        histMaxLat = str(histMaxLat) +"          " + str(image) + "\n"
        print(histMaxLat)
def main():    
    # ITA = image to analyze | Img = Image | # dir = Directory | crop order: bottom, top, left, right | for general data, use img. for pixel info, use ITA
    acptImgFt = (".jpeg", ".png", ".tif")
    reformattedDir = 'C:/Users/aadvi/Desktop/Coding/T62/Post June 2021/Images/Cyl images/Cyl reformatted/'
    csvDir = 'C:/Users/aadvi/Desktop/Coding/T62/Post June 2021/Lat data/t62_cyl_coords.csv'
    i = 0
    imInfo = [reformattedDir, "this will be replaced in two ms", i ]
    crop = [60, 80, 0, 100]
    lat = Jcube_to_csv(geo_csv = csvDir, var='lat', nL=125, nS=133)
    lon = Jcube_to_csv(geo_csv = csvDir, var='lon', nL=125, nS=133)


    list = os.listdir(reformattedDir) 
    number_files = len(os.listdir(reformattedDir))
    print ("Number of files in directory: ",number_files)
    for filename in os.listdir(reformattedDir):

        if (filename.endswith(acptImgFt)): 
            imInfo[1] = filename
            imageFN = str(imInfo[0] + imInfo[1])
            ImageToAnalyze(imageFN, imInfo,crop, csvDir, lat, lon)
            i+=1
            imInfo[2] = i
main()


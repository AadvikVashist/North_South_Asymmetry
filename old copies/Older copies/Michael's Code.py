#iterate through directory
import os
import os.path
from re import X
import statistics
#other
import sys
import time

#Regressions and Plotting
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
latits = []; bands = []
for pic in range(95):
    def shift (csv, image, left, right):
        lat = Jcube_to_csv(geo_csv = csv, var='lat', nL=361, nS=725)#nL=125, nS=133)
        lon = Jcube_to_csv(geo_csv = csv, var='lon', nL=361, nS=725)#nL=125, nS=133)
        ## read cyl image and convert to dtype npfloat32
        im = plt.imread(image)[:,:,0]
        im = im.astype(np.float32)
        ## create high-contrasting (HC) band image (I/F difference columns)
        hc_band = np.empty((361, 725), float)#125, 133), float)
        nsa_lats = []; nsa_lons = []; cols = []
        for col in range(left, right):#im.shape[1]):
            '''## construct column of HC band image
            shift_IF_down  = np.insert(im[:,col], [0,0], [np.nan,np.nan])
            shift_IF_up  = np.concatenate((im[:,col],[np.nan],[np.nan]))
            hc_band[:,col] = (shift_IF_up - shift_IF_down)[1:-1]
            ## subset HC band column b/t 25°S to 15°N
            if_sh = (shift_IF_up-shift_IF_down)[(np.concatenate(([np.nan],lat[:,col],[np.nan])) > -30.0) &\
                                        (np.concatenate(([np.nan],lat[:,col],[np.nan])) < 0.0)]
            lat_sh = (np.concatenate(([np.nan],lat[:,col],[np.nan])))[(np.concatenate(([np.nan],lat[:,col],[np.nan])) > -30.0) &\
                                        (np.concatenate(([np.nan],lat[:,col],[np.nan])) < 0.0)]'''
            ## new code to append correct number of nans for 3 deg latitude shifts
            ## with input of pixel resolution
            pix_res = 0.5
            num_of_nans = int(3.0/pix_res)
            nans = [np.nan]*num_of_nans
            shift_IF_down = np.insert(im[:,col], [0]*num_of_nans, nans)
            #print(im[:,col].flatten().size)#, np.insert(im[:,col], shift, nans))
            shift_IF_up = np.concatenate((im[:,col], nans))
            #shift_IF_down  = np.insert(im[:,col], [0,0], [np.nan,np.nan])
            #shift_IF_up  = np.concatenate((im[:,col],[np.nan],[np.nan]))
            #print((shift_IF_up - shift_IF_down)[num_of_nans:(-num_of_nans)], lat[:,col].size)
            hc_band[:,col] = (shift_IF_up - shift_IF_down)[int(num_of_nans/2):int(-num_of_nans/2)]
            ## subset HC band column b/t 30°S to 0°N
            #hc_band[crop[0]:crop[1],crop[2]:crop[3]]
            if_sh = hc_band[:,col][(lat[:,col] > -30.0) & (lat[:,col] < 0.0)]
            lat_sh = lat[:,col][(lat[:,col] > -30.0) & (lat[:,col] < 0.0)]
            lon_sh = lon[:,col][(lat[:,col] > -30.0) & (lat[:,col] < 0.0)]
            ## apply 6th-order polyfit to I/F brightness profile
            popt, _ = curve_fit(poly6, lat_sh, if_sh)
            ## find and record latitude of minimum I/F in 6th-order polyfit
            nsa_lats.append(lat_sh[poly6(lat_sh, *popt) == np.min(poly6(lat_sh, *popt))][0])
            #nsa_lats.append(lat_sh[poly6(lat_sh, *popt) == np.max(np.abs(poly6(lat_sh, *popt)))][0])
            #nsa_lons.append(lon_sh[poly6(lat_sh, *popt) == np.max(np.abs(poly6(lat_sh, *popt)))][0])
            cols.append(col)
        return nsa_lats
    #plt.hist(shift(d+'t62_cyl_coords.csv', d+'T62.vis.cyl.90.Jcube.tif'), bins=30)
    #plt.ylabel('Count'); plt.xlabel('Latitude')
    #print(np.median(nsa_lats))
    ## T85
    idx = pic#72
    #print(idx)
    d = 'nsa_files/T85_cyl_tifs/'
    im = plt.imread('C:/Users/aadvi/Desktop/Titan Paper/Data/T85/T85.vis.cyl/T85.cyl12.tif')[:,:,0]
    lat = Jcube_to_csv(geo_csv ='C:/Users/aadvi/Desktop/Titan Paper/Data/T85/T85.CSV/T85_geo_cyl.csv', var='lat', nL=361, nS=725)
    lon = Jcube_to_csv(geo_csv = 'C:/Users/aadvi/Desktop/Titan Paper/Data/T85/T85.CSV/T85_geo_cyl.csv', var='lon', nL=361, nS=725)
    data = shift('C:/Users/aadvi/Desktop/Titan Paper/Data/T85/T85.CSV/T85_geo_cyl.csv', 'C:/Users/aadvi/Desktop/Titan Paper/Data/T85/T85.vis.cyl/T85.cyl12.tif', 200, 400)
    '''## T85 Cylindrical projection
    plt.figure(figsize=(10,10));
    plt.imshow(im, cmap="Greys_r"); #plt.colorbar()
    c = plt.contour(lat, levels = np.arange(-90,90+30,30), colors = "r", alpha=0.5)
    plt.clabel(c,c.levels,inline=1,fmt = '%1.0f°N',fontsize=10); plt.show()
    ## T85 High-contrasting band....
    #shift(d+'T85_geo_cyl.csv', d+'T85.cyl73.tif', 200, 400)
    plt.figure(figsize=(10,10)); plt.imshow(data[1],
                                            cmap='Greys_r',vmin=-10,vmax=10);
    #plt.colorbar()
    c = plt.contour(lat, levels = np.arange(-90,90+30,30), colors = "r", alpha=0.7)
    plt.clabel(c,c.levels,inline=1,fmt = '%1.0f°N',fontsize=10)
    c2 = plt.contour(lon, levels = [60,300], colors = "r", alpha=0.7)
    plt.clabel(c2,c2.levels,inline=1,fmt = '%1.0f°E',fontsize=10)
    #shift(d+'T85_geo_cyl.csv', d+'T85.cyl73.tif')
    plt.figure(figsize=(10,10))
    plt.hist(data[0], bins=20, density=True)'''
    arr = np.array(data[0])
    #print(arr)
    #arr = arr[np.where(arr > -12)[0]]
    from scipy.stats import norm
    mean = np.mean(arr)
    variance = np.var(arr)
    sigma = np.sqrt(variance)
    x = np.linspace(min(arr), max(arr), arr.size)
    y = norm.pdf(x, mean, sigma)
    '''plt.plot(x, y, label='old fit')
    from scipy.optimize import leastsq
    fitfunc  = lambda p, x: p[0]*np.exp(-0.5*((x-p[1])/p[2])**2)+p[3]
    errfunc  = lambda p, x, y: (y - fitfunc(p, x))
    init  = [1.5, 0.5, 0.5, 0.5]
    out   = leastsq(errfunc, init, args=(x, y))
    c = out[0]
    plt.plot(x, fitfunc(c, x), c='k', label='better fit?')
    from scipy.stats import weibull_min
    c = 1.79
    plt.plot(x, weibull_min.pdf(x, c),'r-', lw=5, label='weibull_min pdf')
    plt.legend()
    plt.ylabel('Count'); plt.xlabel('Latitude')
    import seaborn as sns
    sns.set_style('darkgrid')
    #fig, ax  = plt.figure()
    sns.histplot(arr,kde=True,bins=10)
    plt.figure()'''
    arr = np.array(shift('C:/Users/aadvi/Desktop/Titan Paper/Data/T85/T85.CSV/T85_geo_cyl.csv', 'C:/Users/aadvi/Desktop/Titan Paper/Data/T85/T85.vis.cyl/T85.cyl12.tif', 200, 400)[0])
    x = np.linspace(min(arr), max(arr), arr.size)
    density = sum(norm(xi).pdf(x) for xi in arr)
    '''plt.hist(arr, bins=10, alpha=0.5)
    plt.fill_between(x, density, alpha=0.5)
    plt.scatter(x[density == np.nanmax(density)][0], density[density == np.nanmax(density)][0],c='r')
    #plt.plot(x, np.full_like(x, -0.1), '|k', markeredgewidth=1)'''
    latits.append(x[density == np.nanmax(density)][0])
    bands.append(pic)
plt.figure()
x = np.linspace(min(latits), max(latits), arr.size)
density = sum(norm(xi).pdf(x) for xi in latits)
plt.hist(latits, bins=10, alpha=0.5)
plt.fill_between(x, density, alpha=0.5)
plt.scatter(x[density == np.nanmax(density)][0], density[density == np.nanmax(density)][0],c='r')
plt.figure(); plt.scatter(bands,latits)
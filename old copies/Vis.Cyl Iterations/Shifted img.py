## import latitude and longitude data/import time
from PIL import Image, ImageStat
import pygsheets
#science/ plotting
import matlab
import pylab
import numpy as np; import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import scipy
from scipy import stats
import warnings
import pyvims
import numpy as np; import pandas as pd
import time
import scipy.misc
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
## 6th-order polynomial
def poly6(x, g, h, i, j, a, b, c):
        return g*x**6+h*x**5+i*x**4+j*x**3+a*x**2+b*x+c
def shift (csv, image, i):
    lat = Jcube_to_csv(geo_csv = csv, var='lat', nL=125, nS=133)
    lon = Jcube_to_csv(geo_csv = csv, var='lon', nL=125, nS=133)
    ## read cyl image and convert to dtype npfloat32 
    im = plt.imread(image)
    im = im.astype(np.float32)
    time.sleep(10)
    
    ## create high-contrasting (HC) band image (I/F difference columns)
    hc_band = np.empty((125, 133), float)
    nsa_lats = []
    for col in range(im.shape[1]):
        ## construct column of HC band image
        shift_IF_down  = np.insert(im[:,col], [0,0], [np.nan,np.nan])
        shift_IF_up  = np.concatenate((im[:,col],[np.nan],[np.nan]))
        hc_band[:,col] = (shift_IF_up - shift_IF_down)[1:-1]
        ## subset HC band column b/t 25°S to 15°N
        if_sh = (shift_IF_up-shift_IF_down)[(np.concatenate(([np.nan],lat[:,col],[np.nan])) > -25.0) &\
                                    (np.concatenate(([np.nan],lat[:,col],[np.nan])) < 15.0)]
        lat_sh = (np.concatenate(([np.nan],lat[:,col],[np.nan])))[(np.concatenate(([np.nan],lat[:,col],[np.nan])) > -25.0) &\
                                    (np.concatenate(([np.nan],lat[:,col],[np.nan])) < 15.0)]
        ## apply 6th-order polyfit to I/F brightness profile
        popt, _ = scipy.optimize.curve_fit(poly6, lat_sh, if_sh)
        ## find and record latitude of minimum I/F in 6th-order polyfit
        nsa_lats.append(lat_sh[poly6(lat_sh, *popt) == np.min(poly6(lat_sh, *popt))][0])
    
    ## image plot to show HC band (can be dark or bright depending on wavelength)
    ## NOTE: vmin, vmax will need to change for each image
    plt.figure(figsize=(10,10)); plt.imshow(hc_band, cmap='gray', vmin=-10,vmax=10)
    plt.colorbar() 
    plt.show()
    img = Image.open(image)
    ITA = img.load()
    width, height = img.size
    if i<10:
        i = "0" + str(i)                 
    else:
        i = str(i) 
    time.sleep(3)
    im = Image.fromarray(hc_band,mode='L')
    matplotlib.image.imsave(( 'C:/Users/aadvi/Desktop/Coding/T62/Frame Images/Cyl images/Cyl shifted/T62.vis.cyl.' + i +".shifted.png"),hc_band,)
    im.show()
    im.save(  'C:/Users/aadvi/Desktop/Coding/T62/Frame Images/Cyl images/Cyl shifted/T62.vis.cyl.' + i +".shifted.png")
url= "C:/Users/aadvi/Desktop/Coding/T62/Frame Images/Cyl images/Cyl reformatted/T62.vis.cyl."
for i in range(0,96, 1): 
    if i<10:
        ImgName = url + "0" + str(i) +  ".Jcube.tif"                      
    else:
        ImgName = url + str(i) +  ".Jcube.tif"                         
    shift('C:/Users/aadvi/Desktop/Coding/T62/Frame Images/Lat data/t62_cyl_coords.csv', ImgName, i)
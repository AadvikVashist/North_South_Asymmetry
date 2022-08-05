# ITA = image to analyze 
# Img = Image
# for general data, use img. for pixel info, use ITA

#Imports.
import numpy as np; import pandas as pd
from PIL import Image, ImageStat
#science/ plotting
import time
import matlab
import pylab
import matplotlib.pyplot as plt
from scipy import stats
import warnings
import pyvims
#Jcube_to_csv program
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
#call the program
lat = Jcube_to_csv(geo_csv = 'C:/Users/aadvi/Desktop/Coding\T62/Frame Images/Lat data/t62_cyl_coords.csv', var = 'lat', nL = 125 , nS = 133 )
#get necessary data.

## Plot np arrays as a check
plt.imshow(lat, vmin = -90, vmax = 90); plt.colorbar()
plt.show()

Jcube_to_csv(geo_csv = 'C:/Users/aadvi/Desktop/Coding/T62/Frame Images/Lat data/t62_cyl_coords.csv', var = 'lat', nL = 125 , nS = 133 )

#determines if image can be opened
def imageCanOpen(image):
    try:
        img = Image.open(image)
    except:
        print("image " + image + " cannot be opened")
        return ("image " + image + "cannot be opened ")
    try:
        ITA = img.load()
    except:
        print("image " + image + " cannot be opened")
        return ("image " + image + "cannot be loaded ")
    returnstatement = 'image "' + image + '" was loaded succesfully.'
    return returnstatement
#Brightness of Image
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
def regressionAnaylsis(bX, bY):
    regressions = {}
    X, Y =  bX, bY
    x = np.array(X)
    y = np.array(Y)
    #linear
    linearRSQ = polyfit( x, y, 1)
    #quadratic
    quadRSQ = polyfit( x, y, 2)
    #cubic
    cubRSQ = polyfit( x, y, 3)
    #quartic
    quartRSQ = polyfit( x, y, 4)
    #quintic
    quintRSQ = polyfit( x, y, 5)
    #sextic
    sextRSQ = polyfit( x, y, 6)
    #Hyperbolic tangent
    #Logistic
    #Logarithmic
    #determines which function should be returned
    return linearRSQ, quadRSQ, cubRSQ, quartRSQ, quintRSQ, sextRSQ

#Main Code   
def ImagetoAnalyze (image):
    print(imageCanOpen(image))
    img = Image.open(image)
    ITA = img.load()
    width, height = img.size
    print(width, height)

    #list to store pixel data
    LatX = []
    pixelRGB = []
    avg = 0
    #average brightness of image
    bright = brightness(image)
    #all RSQ averages.
    linearRSQavg = quadRSQavg = cubRSQavg = quartRSQavg = quintRSQavg= sextRSQavg = colNum= 0
    #Lat Data
    time.sleep(1)
    for ximg in range (0,width,1):
        lat = Jcube_to_csv(geo_csv = 'C:/Users/aadvi/Desktop/Coding\T62/Frame Images/Lat data/t62_cyl_coords.csv', var = 'lat', nL = 125 , nS = 133 )
        for i in range(0, 125,1):
            LatX.append(lat[i,0])
        for yimg in range (0,height,1):  
            rgb = ITA[ximg,yimg]
            pixelRGB.append(rgb)
        linearRSQ, quadRSQ, cubRSQ, quartRSQ, quintRSQ, sextRSQ = regressionAnaylsis(LatX, pixelRGB)
        #Averages.  
        linearRSQavg += linearRSQ
        cubRSQavg += cubRSQ
        quadRSQavg += quadRSQ
        quartRSQavg += quartRSQ
        quintRSQavg += quintRSQ
        sextRSQavg += sextRSQ
        colNum += 1
        
    plt.plot(LatX,pixelRGB, 'o')
        
    print(colNum, " \n" , linearRSQavg / colNum, " \n",quadRSQavg / colNum," \n",cubRSQavg / colNum , " \n",quartRSQavg / colNum, " \n", quintRSQavg / colNum, " \n", sextRSQavg / colNum)
    #run the regresion line, and determine the best possible equation.  
    plt.show()    
    img.show()

#regression paramters are the actual data, as well as the average of the data set. This is used to determine which points of the line are necessary to graph
for i in range(0,100, 1): 
    if i<=10:
        ImgName = "C:/Users/aadvi/Desktop/Coding/T62/Frame Images/Vis.Cyl/T62.vis.cyl.0" + str(i) +  ".png"                      
        ImagetoAnalyze(ImgName)
        time.sleep(10)
    else:
        ImgName = "C:/Users/aadvi/Desktop/Coding/T62/Frame Images/Vis.Cyl/T62.vis.cyl." + str(i) +  ".png"                      
        ImagetoAnalyze(ImgName)
        time.sleep(10)

    



#-90,90 for the x balues in model . Max Y value = data need to be collected. store data in lists. 
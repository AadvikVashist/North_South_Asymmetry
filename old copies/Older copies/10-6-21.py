#iterate through directory
import os
import os.path
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


#fix runtime errors warnings.filterwarnings("ignore")
#progress bar
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
#image brightnes
def brightness( im_file ):
    im = Image.open(im_file).convert('L')
    stat = ImageStat.Stat(im)
    return stat.mean[0]
#N/S Boundary detection fits
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
#analysis code
def shift (csv, image, lat, lon, crop):
    ## read cyl image data, and convert to float 32 type for analysis
    im = plt.imread(image)
    imO = Image.open(image)
    imW, imH = imO.size
    im = im.astype(np.float32)
    ## create high-contrasting band image (I/F difference columns)
    hc_band = np.empty((imH,imW), float)
    nsa_lats = []; nsa_lons = []
    cols = []
    for col in range(imW): # 133 = image width
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
    #lats = csv
    lats = lat.tolist()
    latlist = []
    x=0
    plottedlat = []
    # makes 2d array into 1d array
    for i in lats:
        latlist.append(lat[x,0])
        x+=1
    # connect point value to lat value
    for i in range(len(nsa_lats)):
        for x in range(len(latlist)):
            if nsa_lats[i] == latlist[x]:
                plottedlat.append(x)
    
    #plotted lat = real values
    
    try: 
        slope, intercept, r, p, se = stats.linregress(cols, plottedlat)
    except:
        return(0, 0, 0, 0, False, 0)
    deviation = np.std(plottedlat)
    #line: converts lat back into plottable points
    f = lambda x:  intercept + slope*x-crop[0]
    z = lambda x:  x - crop[ 0]
    #plottable points
    yaxis = [f(x) for x in cols]
    points = [z(x) for x in plottedlat]
    #plot
    """
    plt.figure(figsize=(10,10)); plt.imshow(hc_band[crop[0]:crop[1], crop[2]:crop[3]], cmap='Greys_r',
                                        vmin=-10,vmax=10)
    plt.plot(cols, yaxis, 'green', label='fitted line')
    plt.plot(cols, points,color = "red",linewidth = 2)
    plt.scatter(cols, points,color = "blue")
    plt.colorbar()
    plt.show() 
    """
    #histogram
    avg = np.mean(points)
    var = np.var(points)
    # shape of the fitted Gaussian.
    if len(points) < 1 or var <= 1: 
        return(0, 0, 0, 0, False, 0)
    pdf_x = np.linspace(np.min(points),np.max(points),100)
    pdf_y = 1.0/np.sqrt(2*np.pi*var)*np.exp(-0.5*(pdf_x-avg)**2/var)

    #histogram plotting :
    plt.figure()
    plt.hist(points,30,density=True)
    plt.plot(pdf_x,pdf_y,'k--')
    #conversion from point back to lat
    Histmax = np.argmax(pdf_y) 
    preLat = int(pdf_x[Histmax] + crop[0]) #crop[0] comes from lambda z
    histOut = lat[preLat][0]
    # due dilligence
    plt.close('all') 
    return nsa_lats, nsa_lons, hc_band, histOut, True, deviation
def ImageToAnalyze (image, directoryList, cropParam, lat, lon):
    #print(image)
    nsa_lats, nsa_lons, uncroppedimgdata, histMaxLat, computable, deviation  = shift(directoryList[~0], image,lat, lon,cropParam)
    folder_path = directoryList[4] + "Results"
    #generates folders
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    #generates and opens all txt files
    latResults = folder_path + "/Lat.txt"
    deviationResults = folder_path + "/Deviations.txt"
    filenameResults = folder_path + "/Filenames.txt"
    anomalyResults = folder_path + "/Anomaly.txt"
    if directoryList[2] > 0:
        openType = "a+"
    else:
        openType = "w"
    latFile = open(latResults, openType)
    deviationFile = open(deviationResults, openType)
    anomalyFile = open(anomalyResults,openType)
    filenameFile = open(filenameResults, openType)
    #writes to txt file
    print(computable)
    if computable == True:
        latFile.write(str(histMaxLat) + "\n")
        deviationFile.write(str(deviation) + "\n")
        filenameFile.write(image + "\n")
    elif computable == False:
        anomalyFile.write(image + "\n")
    filenameFile.close()
    anomalyFile.close()
    deviationFile.close()
    latFile.close()
def main(DirectoryFolder, imgDirectoryFolder):    
    i = 0
    # ITA = image to analyze | Img = Image | # dir = Directory | crop order: bottom, top, left, right | for general data, use img. for pixel info, use ITA
    imgFileTypes = (".jpeg", ".png", ".tif")
    csvDirectory = 'C:/Users/aadvi/Desktop/Titan Paper/Data/Latitude data/t62_cyl_coords.csv'
    fileInfo = [imgDirectoryFolder, "", i, csvDirectory, DirectoryFolder]
    #crop of image for plotting + lines
    crop = [0, 361, 0, 725]
    #csv data 
    lat = Jcube_to_csv(geo_csv = csvDirectory, var='lat', nL=125, nS=133)
    lon = Jcube_to_csv(geo_csv = csvDirectory, var='lon', nL=125, nS=133)
    #find directory
    list = os.listdir(imgDirectoryFolder) 
    number_files = len(os.listdir(imgDirectoryFolder))
    print ("Number of files in directory: ",number_files)
    #iterates over all files in directory
    for filename in os.listdir(imgDirectoryFolder):
        i = 0
        if (filename.endswith(imgFileTypes)): 
            fileInfo[1] = filename
            imageFN = str(fileInfo[0] + fileInfo[1])
            ImageToAnalyze(imageFN, fileInfo,crop, lat, lon)
            i+=1
            fileInfo[2] = i
bigFolder = 'C:/Users/aadvi/Desktop/Titan Paper/Data/T85/'
imageFolder = 'C:/Users/aadvi/Desktop/Titan Paper/Data/T85/T85.vis.cyl/'
main(bigFolder, imageFolder )

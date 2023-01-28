from data import data
import matplotlib.pyplot as plt
import os
import os.path as path
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
class Tilt: 
    def __init__(self, directory, Tdataset, dataInfo):
        self.allDatasets = []
        self.NSA =[]
        self.NS_Flux_Ratio = []
        self.dev = []
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
        self.dev.append(((self.data[1])[1:-1]).astype(np.float64))
    def aTiltPlot(self):
            tiltDatasets = [3, 5]
            bands = [74, 89]
            size = [12, 8]
            columns = [400, 600]
            xLabel = 'Longitude (°)'
            yLabel = 'North South Boundary Latitude (°)'
            axisFontSize = 12
            Title = "Axis Tilt of NSA found in the "
            titleFontSize = 10
            lineWidth = 2
            dataPointStyle = ','
            angle = "The NSA is "
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
    def bTiltPlot(self):
            tiltDatasets = [3, 5]
            bands = [50, 75]
            size = [12, 8]
            columns = [0, 600]
            xLabel = 'Longitude (°)'
            yLabel = 'North South Boundary Latitude (°)'
            axisFontSize = 12
            Title = "Axis Tilt of NSA found in the "
            titleFontSize = 10
            lineWidth = 2
            dataPointStyle = ','
            angle = ["The NSA is located at ", " with an angle of "]
            self.wavelengths()
            for i in self.allDatasets:
                self.datasetRead(i)
            movingSmooth = 5
            figures = []
            purpose = ["tilt",bands,columns, "", movingSmooth]
            for i in range(len(tiltDatasets)):
                dataTdataset = self.Tdataset[tiltDatasets[i]]
                a = data(self.directory, dataTdataset, self.shiftDegree, purpose)  
                for bander in range(len(bands)):  
                    xPlot = list(a.band[bander][1])
                    yPlot = list(a.band[bander][2])
                    yLine = [x*a.band[bander][3][0]+a.band[bander][3][1] for x in xPlot]
                    plt.rcParams["font.family"] = 'monospace'
                    plt.rcParams["font.weight"] = 'light'
                    fig = plt.figure(num = 1, figsize = size, dpi = 80, facecolor='w', edgecolor = 'k')
                    plt.xlabel(xLabel, fontsize = axisFontSize);plt.ylabel(yLabel, fontsize = axisFontSize)
                    if len(tiltDatasets) > 1:
                        title = Title + self.Tdataset[tiltDatasets[i]][0] + " dataset at wavelength " + str(self.wavelength[74]) + 'µm'
                        plt.title(title, fontsize = titleFontSize*1.25)
                    else:
                        plt.title(Title, fontsize = titleFontSize*1.25)
                    plt.plot(xPlot, yPlot, linewidth=lineWidth, marker = dataPointStyle, color = (0,0,0,1))
                    plt.plot(xPlot, yLine, linewidth=lineWidth, marker = dataPointStyle, color = (1,0,0,1))
                    plt.figtext(0.24,0.18, angle[0] + str(self.NSA[i][bander]) + angle[1] + str(self.angle(a.band[0][3][0])) + "°")
                    plt.ylim(-30,30)
                    print(str(a.band[bander][5]))
                    plt.grid(axis = 'y')
                    plt.show() 
                    figures.append(fig)
                print("new")
            return figures
    def cTiltPlot(self):
            tiltDatasets = [3, 5]
            bands = [50, 75]
            size = [16,16]
            columns = [0, 600]
            xLabel = 'Longitude (°)'
            yLabel = 'North South Boundary Latitude (°)'
            axisFontSize = 15
            Title = "Axis Tilt of NSA found in the "
            titleFontSize = 12
            tickSize = 15
            lineWidth = 2
            dataPointStyle = ','
            angle = ["The NSA is located at ", "°S with an angle of "]
            self.wavelengths()
            for i in self.allDatasets:
                self.datasetRead(i)
            movingSmooth = 5
            figures = []
            purpose = ["tilt",bands,columns, "", movingSmooth]
            for i in range(len(tiltDatasets)):
                dataTdataset = self.Tdataset[tiltDatasets[i]]
                a = data(self.directory, dataTdataset, self.shiftDegree, purpose)  
                for bander in range(len(bands)):  
                    xPlot = list(a.band[bander][1])
                    yPlot = list(a.band[bander][2])
                    yLine = [x*a.band[bander][3][0]+a.band[bander][3][1] for x in xPlot]
                    plt.rcParams["font.family"] = 'monospace'
                    plt.rcParams["font.weight"] = 'light'
                    fig = plt.figure(figsize = size)
                    plt.xticks(size = tickSize); plt.yticks(size = tickSize)
                    plt.xlabel(xLabel, fontsize = axisFontSize);plt.ylabel(yLabel, fontsize = axisFontSize)
                    if len(tiltDatasets) > 1:
                        title = Title + self.Tdataset[tiltDatasets[i]][0] + " dataset at wavelength " + str(self.wavelength[74]) + 'µm'
                        plt.title(title, fontsize = titleFontSize*1.25)
                    else:
                        plt.title(Title, fontsize = titleFontSize*1.25)
                    plt.plot(xPlot, yPlot, linewidth=lineWidth, marker = dataPointStyle, color = (0,0,0,1))
                    plt.plot(xPlot, yLine, linewidth=lineWidth, marker = dataPointStyle, color = (1,0,0,1))
                    plt.figtext(0.5,0.18, angle[0] + str(self.NSA[i][bander]) + angle[1] + str(self.angle(a.band[0][3][0])) + "°", size =15, horizontalalignment='center')
                    plt.ylim(-30,30)
                    print(str(a.band[bander][5]))
                    plt.grid(axis = 'y')
                    plt.show() 
                    figures.append(fig)
                print("new")
            return figures
    def angle(self, slope):
        return 180/np.pi*np.arctan(slope)
    def linearRegress(self, x, y):
        return np.polyfit(x,y,1, cov = True)
    def NSATilt(self, x, y):
        function, V = self.linearRegress(x,y)
        x=np.array(x, dtype = "float64")
        ys =  x*function[0]+function[1]
        a = r2_score(y, ys)
        print(self.angle(np.sqrt(V[0][0])), np.sqrt(V[1][1]))

        return function, a, ys, V
    def image(self, datasets, band):  
        datasets = datasets[0]
        currentfiles = sorted([self.directory[0] + '/' + datasets + '/' + self.directory[3] + '/' + e for e in os.listdir(self.directory[0] + '/' + datasets + '/' + self.directory[3])])[band]
        try: #open image arrays
            im = plt.imread(currentfiles)[:,:,0]
        except:
            im = plt.imread(currentfiles)[:,:]
        return im
    def dTiltPlot(self):
            #info
            xLabel = 'Longitude (°)'
            yLabel = 'North South Boundary Latitude (°)'
            axisFontSize = 15
            Title = "Axis Tilt of NSA found in the "
            titleFontSize = 12
            tickSize = 15
            lineWidth = 2
            dataPointStyle = ','
            angle = ["The NSA is located at ", "°S with an angle of "]
            tiltDatasets = [3, 5]
            bands = [74,89]
            size = [16,16]
            self.wavelengths()
            for i in self.allDatasets:
                self.datasetRead(i)
            movingSmooth = 1
            figures = []
            #Cropping image and datapoints
            columnRange = [600,725, 0,200]
            columns = [*range(columnRange[0],columnRange[1])]
            b = [*range(columnRange[2],columnRange[3])]
            columns.extend(b)
            purpose = ["tilt",bands,columns, "", movingSmooth]
            for i in range(len(tiltDatasets)):
                dataTdataset = self.Tdataset[tiltDatasets[i]]
                a = data(self.directory, dataTdataset, self.shiftDegree, purpose) #columns,lon,lats
                for currentBand in range(len(bands)):
                    plt.rcParams["font.family"] = 'monospace'
                    plt.rcParams["font.weight"] = 'light'
                    #get image with band
                    images = self.image(dataTdataset, currentBand)
                    imageA = images[:,[*range(columnRange[0],columnRange[1])]]
                    imageB = images[:,[*range(columnRange[2],columnRange[3])]]
                    croppedImage = np.concatenate((imageA,imageB), axis = 1)
                    """
                    plt.imshow(images, cmap = 'Greys')
                    plt.show()
                    plt.imshow(croppedImage, cmap = 'Greys')
                    plt.show()
                    """
                    #data
                    columns = a.band[currentBand][0]
                    lon_shTilt = [i - 180 for i in a.band[currentBand][1]]
                    nsa_lats = a.band[currentBand][2]
                    #split image to visible spectrum
                    column = [columns[columnRange[1]-columnRange[0]::],columns[0:columnRange[1]-columnRange[0]]]
                    nsa_lat = [nsa_lats[columnRange[1]-columnRange[0]::],nsa_lats[0:columnRange[1]-columnRange[0]]]
                    lon_sh = [lon_shTilt[columnRange[1]-columnRange[0]::],lon_shTilt[0:columnRange[1]-columnRange[0]]]
                    x = np.concatenate((lon_sh[1][0:-2], lon_sh[0][2::]))
                    y = np.concatenate((nsa_lat[1][0:-2], nsa_lat[0][2::]))
                    for xyz in range(len(x)):
                        if x[xyz] <= 0:
                            x[xyz] += 360
                    xy = list(zip(x, y))
                    sorted_pairs = sorted(xy)
                    tuples = zip(*sorted_pairs)
                    x, y = [ list(tuple) for tuple in  tuples]
                    d,e,f,g = self.NSATilt(x,y) # function, r^2, fitted vals (don't use), covariance matrix
                    #lines
                    fig = plt.figure(figsize = size)
                    firstLine = [d[0]*0 + d[1], d[0]*(columnRange[1]-columnRange[0]) + d[1]]
                    secondLine = [firstLine[1], firstLine[0] + d[0]*(columnRange[3]-columnRange[2])]
                    plt.plot([min(lon_sh[0][2::]), max(lon_sh[0][2::])],firstLine, color = (0.2,0.75,1), linewidth = lineWidth)
                    plt.plot([min(lon_sh[1][0:-2]), max(lon_sh[1][0:-2])],secondLine, color = (0.2,0.75,1), linewidth = lineWidth)
                    #plot data
                    plt.plot(lon_sh[0][2::], nsa_lat[0][2::], color = (1,0,0,1), linewidth = lineWidth)
                    plt.plot(lon_sh[1][0:-2], nsa_lat[1][0:-2], color = (1,0,0,1), linewidth = lineWidth)
                    #show image band
                    plt.imshow(np.flip(images, axis = 1), cmap = 'Greys', extent = (-180,180,-90,90))
                    #ticks
                    plt.xticks(np.arange(-180,180+30, 30), size = tickSize)
                    plt.yticks(np.arange(-90,90+15, 15), size = tickSize)
                    print(self.angle(np.sqrt(g[0][0])))
                    plt.figtext(0.5,0.25, "The NSA is located at"  + str(round(self.NSA[i][currentBand],3)) + " ± " + str(round(self.dev[i][currentBand],3)) + "°S  with an angle of" + str(round(self.angle(d[0]),3)) + "°", size =12, horizontalalignment='center')
                    plt.xlabel(xLabel, fontsize = axisFontSize);plt.ylabel(yLabel, fontsize = axisFontSize)
                    """if len(tiltDatasets) > 1:
                        title = Title + self.Tdataset[tiltDatasets[i]][0] + " dataset at wavelength " + str(self.wavelength[74]) + 'µm'
                        plt.title(title, fontsize = titleFontSize*1.25)
                    else:
                        plt.title(Title, fontsize = titleFontSize*1.25)
                    plt.plot(xPlot, yPlot, linewidth=lineWidth, marker = dataPointStyle, color = (0,0,0,1))
                    plt.plot(xPlot, yLine, linewidth=lineWidth, marker = dataPointStyle, color = (1,0,0,1))
                    plt.figtext(0.5,0.18, angle[0] + str(self.NSA[i][currentBand]) + angle[1] + str(self.angle(a.band[0][3][0])) + "°", size =15, horizontalalignment='center')
                    plt.ylim(-30,30)
                    print(str(a.band[currentBand][5]))
                    plt.grid(axis = 'y')
                    plt.show() 
                    """
                    plt.show()
                    figures.append(fig)
                print("new")
            return figures
    def eTiltPlot(self):
            
            #info
            xLabel = 'Longitude (°)'
            yLabel = 'North South Boundary Latitude (°)'
            axisFontSize = 15
            Title = "Axis Tilt of NSA found in the "
            titleFontSize = 12
            tickSize = 15
            lineWidth = 2
            dataPointStyle = ','
            angle = ["The NSA is located at ", "°S with an angle of "]
            tiltDatasets = [5]
            bands = [73,89]
            size = [16,16]
            self.wavelengths()
            for i in self.allDatasets:
                self.datasetRead(i)
            movingSmooth = 1
            figures = []
            #Cropping image and datapoints
            columnRange = [600,725, 0,200]
            columns = [*range(columnRange[0],columnRange[1])]
            b = [*range(columnRange[2],columnRange[3])]
            columns.extend(b)
            purpose = ["tilt",bands,columns, "", movingSmooth]
            yTicks = np.arange(-90,90+15, 15)
            yTick = [str(i) + "°N" if i >= 0 else str(abs(i)) + "°S" for i in list(yTicks)] 
            xTicks = [-180,-120,-60,0,60,120,180]
            xTick = ["0°E","60°E","120°E","180°","120°W","60°W","0°W"]
            t67 = ["/Users/aadvik/Downloads/t67_repeat/0.png","/Users/aadvik/Downloads/t67_repeat/1.png"]

            # xTick = [str(i) + "°E" if i >= 0 else str(abs(i)) + "°W" for i in list(xTick)] 
            try:
                try: 
                    a = yTick.index("0.0°N")
                except:
                    a = yTick.index("0°N")
                yTick[a] = "0°"
            except:
                pass
            for i in range(len(tiltDatasets)):
                dataTdataset = self.Tdataset[tiltDatasets[i]]
                a = data(self.directory, dataTdataset, self.shiftDegree, purpose) #columns,lon,lats
                for currentBand in range(len(bands)):
                    print(self.Tdataset[tiltDatasets[i]][0], self.wavelength[bands[currentBand]])
                    plt.rcParams["font.family"] = 'monospace'
                    plt.rcParams["font.weight"] = 'light'
                    #get image with band
                    images = self.image(dataTdataset, bands[currentBand])
                    imageA = images[:,[*range(columnRange[0],columnRange[1])]]
                    imageB = images[:,[*range(columnRange[2],columnRange[3])]]
                    croppedImage = np.concatenate((imageA,imageB), axis = 1)
                    #data
                    columns = a.band[currentBand][0]
                    lon_shTilt = [i - 180 for i in a.band[currentBand][1]]
                    nsa_lats = a.band[currentBand][2]
                    #split image to visible spectrum
                    column = [columns[columnRange[1]-columnRange[0]::],columns[0:columnRange[1]-columnRange[0]]]
                    nsa_lat = [nsa_lats[columnRange[1]-columnRange[0]::],nsa_lats[0:columnRange[1]-columnRange[0]]]
                    lon_sh = [lon_shTilt[columnRange[1]-columnRange[0]::],lon_shTilt[0:columnRange[1]-columnRange[0]]]
                    x = np.concatenate((lon_sh[1][0:-2], lon_sh[0][2::]))
                    y = np.concatenate((nsa_lat[1][0:-2], nsa_lat[0][2::]))
                    for xyz in range(len(x)):
                        if x[xyz] <= 0:
                            x[xyz] += 360
                    xy = list(zip(x, y))
                    sorted_pairs = sorted(xy)
                    tuples = zip(*sorted_pairs)
                    x, y = [ list(tuple) for tuple in  tuples]
                    d,e,f,g = self.NSATilt(x,y) # function, r^2, fitted vals (don't use), covariance matrix
                    #lines
                    fig = plt.figure(figsize = size)
                    firstLine = [d[0]*0 + d[1], d[0]*(columnRange[1]-columnRange[0]) + d[1]]
                    secondLine = [firstLine[1], firstLine[0] + d[0]*(columnRange[3]-columnRange[2])]
                    plt.plot([min(lon_sh[0][2::]), max(lon_sh[0][2::])],firstLine, color = (0.2,0.75,1), linewidth = lineWidth)
                    plt.plot([min(lon_sh[1][0:-2]), max(lon_sh[1][0:-2])],secondLine, color = (0.2,0.75,1), linewidth = lineWidth)
                    #plot data
                    plt.plot(lon_sh[0][2::], nsa_lat[0][2::], color = (1,0,0,1), linewidth = lineWidth)
                    plt.plot(lon_sh[1][0:-2], nsa_lat[1][0:-2], color = (1,0,0,1), linewidth = lineWidth)
                    #show image band
                    #ticks
                    try: #open image arrays
                        images = plt.imread(t67[currentBand])[:,:,0]
                    except:
                        images = plt.imread(t67[currentBand])[:,:]
                    plt.imshow(np.flip(images, axis = 0), cmap = 'Greys_r', extent = (-180,180,-90,90))

                    print("angle is " , self.angle(np.sqrt(g[0][0])))
                    plt.xticks(ticks = xTicks, labels = xTick, size = tickSize*1.25)
                    plt.yticks(ticks = yTicks, labels = yTick, size = tickSize*1.25)
                    
                    # plt.figtext(0.51,0.8, "The NSA is located at "  + str(round(self.NSA[i][currentBand],1)) + "°S ± " + str(round(self.dev[i][currentBand],1)) + " with an angle of " + str(round(self.angle(d[0]),1)) + "°" + " in the " + dataTdataset[0] + " flyby at " + str(np.round(self.wavelength[bands[currentBand]],3)) + "µm", size = 19, color = (1,1,1,1), horizontalalignment='center')
                    plt.xlabel(xLabel, fontsize = axisFontSize);plt.ylabel(yLabel, fontsize = axisFontSize)
                    
                    # plt.show()
                    figures.append(fig)
                print("new")
            return figures
import os
import os.path as path
import matplotlib.pyplot as plt
import matplotlib.dates as date
import numpy as np
import pandas as pd
from data import data
import math
import re
from datetime import datetime
class ComparisonToRoman:
    def __init__(self, directory, Tdataset, shiftDegree):
        self.allDatasets = []
        self.NSA =[]
        self.NS_Flux_Ratio = []
        self.directory = directory
        self.Tdataset = Tdataset
        self.shiftDegree = shiftDegree
        self.wavelengths()
        self.createpltureFolder()
        self.createFileFolder()
        for self.i in Tdataset:
            self.allDatasets.append((self.directory[0] + "/" + self.directory[7] + "/" + self.i[0] + self.directory[5]))
    def createpltureFolder(self):
        folderPath = self.directory[0] + "/" + self.directory[8][0]
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
        self.resultsFolder = folderPath    
    def createFileFolder(self):
        folderPath = self.directory[0] + "/" + self.directory[8][0] + "/" + self.directory[8][3]
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
            row = rowOne.index(i[0])
            self.dates.append(date[row,2])
    def datasetRead(self, x):
        self.data = np.array(pd.read_csv(x, header = None))
        self.NSA.append(((self.data[0])[1:-1]).astype(np.float64))
        self.NS_Flux_Ratio.append(((self.data[2])[1:-1]).astype(np.float64))
    def roman(self):
        self.roman = date = np.array(pd.read_csv(self.directory[0] + '/' + self.directory[1] + '/' + self.directory[9], header = None))
    def aComparison(self):
        self.datasetDates()
        subsetWave = (self.wavelength >= 0.8) & (self.wavelength <= 0.9)
        self.wavelength = self.wavelength[subsetWave]
        self.allDatasets = self.allDatasets[0:3] # Ta T8 T31
        self.Tdataset = self.Tdataset[0:3] # Ta T8 T31
        self.roman()
        self.romanLats = self.roman[4][1::]
        import re
        self.romanLats = [[[float(x.strip()) for x in (i.split(' ± '))] for i in (list(self.romanLats[:]))][i] for i in [0,8,11]] #images 0, 8, and 11
        self.romanDates = self.roman[1][1::]
        self.romanDates = [(self.romanDates[i]).strip() for i in [0,8,11]]
        for i in range(len(self.allDatasets)):
            self.datasetRead(self.allDatasets[i])
            self.NSA[i] = self.NSA[i][subsetWave]
        title = "Comparison to Roman et al. "
        xLabel = "Wavelength (µm)"
        yLabel = "North South Boundary Latitude (°) \n"
        size = [12,16]
        xLim = [0.8,0.9]
        yLim = [-20,0]
        cmapMin = 0.1
        cmapMax = 1
        cMap = "hsv"
        darken = 0.5
        axisFontSize = 10
        titleFontSize = 10
        legendLocation = 4
        legendFontSize = 9
        lineWidth = 2
        xtickCount = 20
        yTickCount = 10
        dataPointStyle = ","
        lineStyles = ["solid", "solid", "solid"]
        grid = 1
        subplotName = [[0.035,0.75],[0.035,0.75],[0.035,0.75],[0.035,0.75],[0.035,0.75],[0.035,0.75],[0.035,0.75],[0.035,0.75],[0.035,0.75],[0.038,0.79],[0.038,0.75],[0.038,0.75]]
        plt.rcParams["font.family"] = 'times'
        plt.rcParams["font.weight"] = 'light'
        xTicks = np.arange(xLim[0], xLim[1]+(xLim[1]-xLim[0])/xtickCount,(xLim[1]-xLim[0])/xtickCount)
        yTicks = np.arange(yLim[0], yLim[1]+(yLim[1]-yLim[0])/yTickCount,(yLim[1]-yLim[0])/yTickCount)
        cMap = plt.cm.get_cmap(cMap)
        try:
            colors = np.arange(cmapMin,(cmapMax+((cmapMax-cmapMin)/len(self.Tdataset))),(cmapMax-cmapMin)/len(self.Tdataset))
        except: 
            colors = [cmapMax for i in range(len(self.Tdataset))]
        print(cMap(colors[0]))
        fig, axs = plt.subplots(nrows = len(self.Tdataset), sharex='all', sharey='all', squeeze = False, figsize = size)
        fig.tight_layout(pad = 1, rect =(0,1.2,.8,1))
        fig.subplots_adjust(hspace = 0.4)
        
        axs = axs.ravel()
        for i in range(len(self.Tdataset)):
            axs[i].set_yticks(yTicks)
            axs[i].set_xlim(xLim)
            axs[i].set_ylim(yLim)
            axs[i].minorticks_on()
            axs[i].yaxis.set_tick_params(labelleft=True)
            axs[i].grid(axis = 'y')
            axs[i].xaxis.set_tick_params(labelbottom=True)
            if i == 1:
                axs[i].set_ylabel(yLabel, fontsize = axisFontSize)
            else:
                axs[i].set_ylabel("")
            axs[i].plot((-20,20), 2*[np.mean(self.NSA[i])], color = ((cMap(colors[i]))[0]*darken,(cMap(colors[i]))[1]*darken,(cMap(colors[i]))[2]*darken,1), linewidth=lineWidth, marker = dataPointStyle, label = self.Tdataset[i][0] + " dataset")
            axs[i].fill_between((-30,10), y1 = -(self.romanLats[i][0] + self.romanLats[i][1]) , y2= -(self.romanLats[i][0] - self.romanLats[i][1]),
                            hatch='///', zorder=2, color = ((cMap(colors[i]))[0]*darken,(cMap(colors[i]))[1]*darken,(cMap(colors[i]))[2]*darken, 0.25), label = self.romanDates[i])
            axs[i].legend(loc = 'upper right')
        axs[i].set_xlabel(xLabel, fontsize = axisFontSize)
        fig.suptitle(title)
        plt.show()
        #lines_labels = [ax.get_legend_handles_labels() for ax in plt.axes]
        #lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        #plt.legend(fontsize = legendFontSize, loc = legendLocation, markerscale = 2, framealpha = 0, frameon = False, edgecolor = 'white', handlelength = 0)
        return fig
    def bComparison(self):
        self.datasetDates()
        subsetWave = (self.wavelength >= 0.75) & (self.wavelength <= 0.95)
        self.wavelength = self.wavelength[subsetWave]
        self.allDatasets = self.allDatasets[0:3] # Ta T8 T31
        self.Tdataset = self.Tdataset[0:3] # Ta T8 T31
        self.roman()
        self.romanLats = self.roman[4][1::]
        self.dates = self.dates[0:3]
        import re
        self.romanLats = [[[float(x.strip()) for x in (i.split(' ± '))] for i in (list(self.romanLats[:]))][i] for i in [0,8,11]] #images 0, 8, and 11
        self.romanDates = self.roman[1][1::]
        self.romanDates = [(self.romanDates[i]).strip() for i in [0,8,11]]
        for i in range(len(self.allDatasets)):
            self.datasetRead(self.allDatasets[i])
            self.NSA[i] = self.NSA[i][subsetWave]
        title = "Comparison to Roman et al. "
        xLabel = "Wavelength (µm)"
        yLabel = "North South Boundary Latitude (°) \n"
        size = [12,16]
        xLim = [0.8,0.9]
        yLim = [-20,0]
        cmapMin = 0.1
        cmapMax = 1
        cMap = "hsv"
        darken = 0.5
        axisFontSize = 10
        titleFontSize = 10
        legendLocation = 4
        legendFontSize = 9
        lineWidth = 2
        xtickCount = 20
        yTickCount = 10
        dataPointStyle = ","
        lineStyles = ["solid", "solid", "solid"]
        grid = 1
        subplotName = [[0.035,0.75],[0.035,0.75],[0.035,0.75],[0.035,0.75],[0.035,0.75],[0.035,0.75],[0.035,0.75],[0.035,0.75],[0.035,0.75],[0.038,0.79],[0.038,0.75],[0.038,0.75]]
        plt.rcParams["font.family"] = 'times'
        plt.rcParams["font.weight"] = 'light'
        xTicks = np.arange(xLim[0], xLim[1]+(xLim[1]-xLim[0])/xtickCount,(xLim[1]-xLim[0])/xtickCount)
        yTicks = np.arange(yLim[0], yLim[1]+(yLim[1]-yLim[0])/yTickCount,(yLim[1]-yLim[0])/yTickCount)
        cMap = plt.cm.get_cmap(cMap)
        try:
            colors = np.arange(cmapMin,(cmapMax+((cmapMax-cmapMin)/len(self.Tdataset))),(cmapMax-cmapMin)/len(self.Tdataset))
        except: 
            colors = [cmapMax for i in range(len(self.Tdataset))]
        print(cMap(colors[0]))
        fig = plt.figure(figsize = size)
        fig.tight_layout(pad = 1, rect =(0,1.2,.8,1))
        fig.subplots_adjust(hspace = 0.4)
        plt.xlabel(xLabel, fontsize = axisFontSize)
        plt.ylabel(yLabel, fontsize = axisFontSize)
        plt.xlim(xLim)
        plt.ylim(yLim)
        plt.xticks(xTicks)
        plt.yticks(yTicks) 
        plt.grid(axis = 'y')
        for i in range(len(self.Tdataset)):
            plt.plot((-30,10), [-(self.romanLats[i][0] + self.romanLats[i][1]),-(self.romanLats[i][0] + self.romanLats[i][1])], color = ((cMap(colors[i]))[0]*darken,(cMap(colors[i]))[1]*darken,(cMap(colors[i]))[2]*darken, 1))
            plt.plot((-30,10), [-(self.romanLats[i][0] - self.romanLats[i][1]),-(self.romanLats[i][0] - self.romanLats[i][1])], color = ((cMap(colors[i]))[0]*darken,(cMap(colors[i]))[1]*darken,(cMap(colors[i]))[2]*darken, 1))
            plt.plot(self.wavelength, self.NSA[i], color = ((cMap(colors[i]))[0]*darken,(cMap(colors[i]))[1]*darken,(cMap(colors[i]))[2]*darken,1), linewidth=lineWidth, marker = dataPointStyle, label = self.Tdataset[i][0] + " dataset taken on " +self.dates[i])
            #plt.fill_between((-30,10), y1 = -(self.romanLats[i][0] + self.romanLats[i][1]) , y2= -(self.romanLats[i][0] - self.romanLats[i][1]),  zorder=2, color = ((cMap(colors[i]))[0]*darken,(cMap(colors[i]))[1]*darken,(cMap(colors[i]))[2]*darken, 0.25), label ="Roman Datasets image taken on " + self.romanDates[i])
        plt.legend(loc = 'upper right')
        plt.title(title)
        plt.show()
        
        return fig
    def cComparison(self):
        self.datasetDates()
        subsetWave = (self.wavelength >= 0.75) & (self.wavelength <= 0.95)
        self.wavelength = self.wavelength[subsetWave]
        self.allDatasets = self.allDatasets[0:3] # Ta T8 T31
        self.Tdataset = self.Tdataset[0:3] # Ta T8 T31
        self.roman()
        self.romanLats = self.roman[4][1::]
        self.dates = self.dates[0:3]
        import re
        self.romanLats = [[[float(x.strip()) for x in (i.split(' ± '))] for i in (list(self.romanLats[:]))][i] for i in [0,8,11]] #images 0, 8, and 11
        self.romanDates = self.roman[1][1::]
        self.romanDates = [(self.romanDates[i]).strip() for i in [0,8,11]]
        for i in range(len(self.allDatasets)):
            self.datasetRead(self.allDatasets[i])
            self.NSA[i] = self.NSA[i][subsetWave]
        title = "Comparison to Roman et al. "
        xLabel = "Wavelength (µm)"
        yLabel = "North South Boundary Latitude (°) \n"
        size = [12,16]
        xLim = [2000,2017]
        yLim = [-20,10]
        cmapMin = 0.1
        cmapMax = 1
        cMap = "hsv"
        darken = 0.5
        axisFontSize = 12
        titleFontSize = 15
        legendLocation = 4
        legendFontSize = 9
        lineWidth = 2
        xtickCount = 20
        yTickCount = 10
        tickSize =10
        dataPointStyle = ","
        lineStyles = ["solid", "solid", "solid"]
        grid = 1
        subplotName = [[0.035,0.75],[0.035,0.75],[0.035,0.75],[0.035,0.75],[0.035,0.75],[0.035,0.75],[0.035,0.75],[0.035,0.75],[0.035,0.75],[0.038,0.79],[0.038,0.75],[0.038,0.75]]
        plt.rcParams["font.family"] = 'times'
        plt.rcParams["font.weight"] = 'light'
        xTicks = np.arange(xLim[0], xLim[1]+(xLim[1]-xLim[0])/xtickCount,(xLim[1]-xLim[0])/xtickCount)
        yTicks = np.arange(yLim[0], yLim[1]+(yLim[1]-yLim[0])/yTickCount,(yLim[1]-yLim[0])/yTickCount)
        yTicks = [str(i) + "N" if i >= 0 else str(i) + "S" for i in yTicks] 
        yTicks[yTicks.index("0N")] = "0"
        cMap = plt.cm.get_cmap(cMap)
        try:
            colors = np.arange(cmapMin,(cmapMax+((cmapMax-cmapMin)/len(self.Tdataset))),(cmapMax-cmapMin)/len(self.Tdataset))
        except: 
            colors = [cmapMax for i in range(len(self.Tdataset))]
        print(cMap(colors[0]))
        fig, axs = plt.subplots(nrows = len(self.Tdataset), sharex='all', sharey='all', squeeze = False, figsize = size)
        fig.tight_layout(pad = 1, rect =(0,1.2,.8,1))
        fig.subplots_adjust(hspace = 0.4)
        axs = axs.ravel()
        for i in range(len(self.Tdataset)):
            axs[i].set_yticks(yTicks, labelsize= tickSize)
            axs[i].set_xticks(xTicks, labelsize= tickSize)
            axs[i].set_xlim(xLim)
            axs[i].set_ylim(yLim)
            axs[i].minorticks_on()
            axs[i].yaxis.set_tick_params(labelleft=True)
            axs[i].grid(axis = 'y')
            axs[i].xaxis.set_tick_params(labelbottom=True)
            if i == 1:
                axs[i].set_ylabel(yLabel, fontsize = axisFontSize)
            else:
                axs[i].set_ylabel("")
            axs[i].plot((-30,10), [-(self.romanLats[i][0] + self.romanLats[i][1]),-(self.romanLats[i][0] + self.romanLats[i][1])], color = ((cMap(colors[i]))[0]*darken,(cMap(colors[i]))[1]*darken,(cMap(colors[i]))[2]*darken, 1))
            axs[i].plot((-30,10), [-(self.romanLats[i][0] - self.romanLats[i][1]),-(self.romanLats[i][0] - self.romanLats[i][1])], color = ((cMap(colors[i]))[0]*darken,(cMap(colors[i]))[1]*darken,(cMap(colors[i]))[2]*darken, 1))
            axs[i].plot(self.wavelength, self.NSA[i], color = ((cMap(colors[i]))[0]*darken,(cMap(colors[i]))[1]*darken,(cMap(colors[i]))[2]*darken,1), linewidth=lineWidth, marker = dataPointStyle, label = self.Tdataset[i][0] + " dataset taken on " +self.dates[i])
            axs[i].fill_between((-30,10), y1 = -(self.romanLats[i][0] + self.romanLats[i][1]) , y2= -(self.romanLats[i][0] - self.romanLats[i][1]),  zorder=2, color = ((cMap(colors[i]))[0]*darken,(cMap(colors[i]))[1]*darken,(cMap(colors[i]))[2]*darken, 0.25), label ="Roman Datasets image taken on " + self.romanDates[i])
            axs[i].legend(loc = 'upper right')
        axs[i].set_xlabel(xLabel, fontsize = axisFontSize)
        fig.suptitle(title)
        plt.show()
        
        return fig
    def dateNumtoFloat(self,x):
        if type(x) is list:
            x = [(date.num2date(i).timetuple().tm_yday-1)/365+ date.num2date(i).year for i in x]
            return x
        else:
            return (date.num2date(x).timetuple().tm_yday-1)/365+ date.num2date(x).year
    def dComparison(self):
        self.datasetDates()
        subsetWave = (self.wavelength >= 0.75) & (self.wavelength <= 0.95)
        #self.wavelength = self.wavelength[subsetWave]
        x = [0,1,2,3,4,5,6,7,8,9,10,11]; Tdata = []; dates = []; self.allData = []; dateObjects = []
        for a in x:
            Tdata.append(self.Tdataset[a][0])# Ta, T92, T108, 278TI
        for i in self.allDatasets:
            for x in Tdata:
                if x in i:
                    self.allData.append(i)
        self.roman()    
        self.romanLats = self.roman[4][1::]
        self.romanLats = [[[float(x.strip()) for x in (i.split(' ± '))] for i in (list(self.romanLats[:]))][i] for i in [0,8,11]] #images 0, 8, and 11
        self.romanDates = self.roman[1][1::]
        self.romanDates = [(self.romanDates[i]).strip() for i in [0,8,11]]
        romans = []
        for i in range(len(self.romanDates)):
            romans.append(datetime.strptime(self.romanDates[i] , '%Y/%j/%H'))
            self.romanDates[i] = date.date2num(datetime.strptime(self.romanDates[i] , '%Y/%j/%H'))
        tNSA = []
        for i in range(len(self.allDatasets)):
            self.datasetRead(self.allDatasets[i])
            self.NSA[i] = self.NSA[i][subsetWave]
            tNSA.append(np.mean(self.NSA[i]))
            dates.append(self.dates[i])
            dateTime = datetime.strptime(self.dates[i], '%m/%d/%Y')
            dateObjects.append(date.date2num(dateTime))
        xLabel = "Year"
        yLabel = "North South Boundary Latitude (°) \n"
        size = [12,16]
        yLim = [-18,12]
        cmapMin = 0
        cmapMax = 1
        cMap = "rainbow"
        darken = 0.75
        axisFontSize = 12
        titleFontSize = 15
        legendLocation = 4
        legendFontSize = 9
        lineWidth = 2
        yTickCount = 10
        tickSize =10
        dataPointStyle = ","
        lineStyles = ["solid", "solid", "solid"]
        grid = 1
        plt.rcParams["font.weight"] = 'light'
        plt.rcParams['font.family'] = 'sans-serif'
        yTicks = np.arange(yLim[0], yLim[1]+(yLim[1]-yLim[0])/yTickCount,(yLim[1]-yLim[0])/yTickCount)
        yTick = [str(i) + "N" if i >= 0 else str(abs(i)) + "S" for i in list(yTicks)] 
        try:
            a = yTick.index("0.0N")
            yTick[a] = "0"
        except:
            pass
        cMap = plt.cm.get_cmap(cMap)
        try:
            colors = np.arange(cmapMin,(cmapMax+((cmapMax-cmapMin)/len(Tdata))),(cmapMax-cmapMin)/(len(Tdata)+len(self.romanDates)))
        except: 
            colors = [cmapMax for i in range(len(self.Tdata))]
        fig = plt.figure(figsize = size)
        #plt.plot_date(dateObjects, tNSA, xdate = True, linestyle='solid', color = (0,0,0,1))
        errorbars = [i[1] for i in self.romanLats]
        self.romanLats  = [-i[0] for i in self.romanLats]
        count = 0
        plt.ylim(yLim)
        plt.yticks(ticks = yTicks, labels = yTick)
        plt.xlabel(xLabel, fontsize = axisFontSize)
        plt.ylabel(yLabel, fontsize = axisFontSize)
        plt.plot(self.dateNumtoFloat(dateObjects), tNSA,marker= ",", color  = (0,0,0,1))
        order = dateObjects.copy(); order.extend(self.romanDates); order = sorted(order)
        a = [0,0]
        for abc in range(len(order)):
            if order[abc] in dateObjects:
                i = a[0]   
                plt.scatter(self.dateNumtoFloat(dateObjects[i]), tNSA[i], linestyle='solid', s = 60, color = (cMap(colors[count])[0]*darken,  cMap(colors[count])[1]*darken, cMap(colors[count])[2]*darken), label = Tdata[i], zorder= 10)
                a[0] +=1
            elif order[abc] in self.romanDates:
                i = a[1]
                plt.errorbar(self.dateNumtoFloat(self.romanDates[i]),self.romanLats[i], errorbars[i], capsize = 6, marker = "o", markersize = 5, elinewidth = 2, capthick = 2, ecolor = (cMap(colors[count])[0]*darken**2,  cMap(colors[count])[1]*darken**2, cMap(colors[count])[2]*darken**2), color = (cMap(colors[count])[0]*darken,  cMap(colors[count])[1]*darken, cMap(colors[count])[2]*darken), label = "Selected Roman Image taken on " + romans[i].strftime("%m/%d/%Y"))
                a[1] +=1
            count+=1
        plt.grid(axis='x')
        plt.legend()
        plt.show()
        """""
        for i in range(len(self.Tdataset)):
            axs[i].set_yticks(ticks = list(yTicks), labels = yTick, fontsize= tickSize)
            axs[i].set_xticks(xTicks, labelsize= tickSize)
            axs[i].set_xlim(xLim)
            axs[i].set_ylim(yLim)
            axs[i].minorticks_on()
            axs[i].yaxis.set_tick_params(labelleft=True)
            axs[i].grid(axis = 'y')
            axs[i].xaxis.set_tick_params(labelbottom=True)
            if i == 1:
                axs[i].set_ylabel(yLabel, fontsize = axisFontSize)
            else:
                axs[i].set_ylabel("")
            axs[i].plot((-30,10), [-(self.romanLats[i][0] + self.romanLats[i][1]),-(self.romanLats[i][0] + self.romanLats[i][1])], color = ((cMap(colors[i]))[0]*darken,(cMap(colors[i]))[1]*darken,(cMap(colors[i]))[2]*darken, 1))
            axs[i].plot((-30,10), [-(self.romanLats[i][0] - self.romanLats[i][1]),-(self.romanLats[i][0] - self.romanLats[i][1])], color = ((cMap(colors[i]))[0]*darken,(cMap(colors[i]))[1]*darken,(cMap(colors[i]))[2]*darken, 1))
            axs[i].plot(self.wavelength, self.NSA[i], color = ((cMap(colors[i]))[0]*darken,(cMap(colors[i]))[1]*darken,(cMap(colors[i]))[2]*darken,1), linewidth=lineWidth, marker = dataPointStyle, label = self.Tdataset[i][0] + " dataset taken on " +self.dates[i])
            axs[i].fill_between((-30,10), y1 = -(self.romanLats[i][0] + self.romanLats[i][1]) , y2= -(self.romanLats[i][0] - self.romanLats[i][1]),  zorder=2, color = ((cMap(colors[i]))[0]*darken,(cMap(colors[i]))[1]*darken,(cMap(colors[i]))[2]*darken, 0.25), label ="Roman Datasets image taken on " + self.romanDates[i])
            axs[i].legend(loc = 'upper right')
        axs[i].set_xlabel(xLabel, fontsize = axisFontSize)
        fig.suptitle(title)
        plt.show()
        """
        
        return fig
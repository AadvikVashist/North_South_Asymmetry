import os
import os.path as path
from tkinter import font
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as plter
import math
class NS_Flux_Ratio:
    def __init__(self, directory, Tdataset, shiftDegree):
        self.allDatasets = []
        self.NSA =[]
        self.NS_Flux_Ratio = []
        self.directory = directory
        self.Tdataset = Tdataset
        self.shiftDegree = shiftDegree
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
    def aNS_Flux(self):
        title = "Seasonal Evolution of Titan's Atmospheric Meridional Brightness Profile"
        xLabel = "Wavelength (??m)"
        yLabel = "N/S"
        size = [24,24]
        xLim = [0.35,1.04]
        yLim = [0.6,2]
        cMap = "hsv"
        cmapMin = 0.9
        cmapMax = 0.9
        axisFontSize = 10
        titleFontSize = 10
        legendLocation = 4
        legendFontSize = 9
        lineWidth = 1.5
        dataPointStyle = ","
        lineStyles = ["solid", "solid", "solid"]
        grid = 1
        subplotName = [[0.035,0.75],[0.035,0.75],[0.035,0.75],[0.035,0.75],[0.035,0.75],[0.035,0.75],[0.035,0.75],[0.035,0.75],[0.035,0.75],[0.038,0.75],[0.038,0.75],[0.038,0.75]]
        self.datasetDates()
        for i in self.allDatasets:
            self.datasetRead(i)
        plt.rcParams["font.family"] = 'times'
        plt.rcParams["font.weight"] = 'light'
        xtickCount = 10
        yTickCount =4
        xTicks = np.arange(xLim[0], xLim[1]+(xLim[1]-xLim[0])/xtickCount,(xLim[1]-xLim[0])/xtickCount)
        yTicks = np.arange(yLim[0], yLim[1]+(yLim[1]-yLim[0])/yTickCount,(yLim[1]-yLim[0])/yTickCount)
        fig, axs = plt.subplots(nrows = math.ceil(len(self.Tdataset)/grid), ncols = grid, sharex='all', sharey='all', squeeze = False, figsize = size)
        cMap = plt.cm.get_cmap(cMap)
        try:
            colors = np.arange(cmapMin,(cmapMax+((cmapMax-cmapMin)/len(self.Tdataset))),(cmapMax-cmapMin)/len(self.Tdataset))
        except: 
            colors = [cmapMax for i in range(len(self.Tdataset))]
        fig.tight_layout(pad = 2, rect =(0,1.5,1,1))
        fig.subplots_adjust(top=0.95, hspace = 0.8)
        axs = axs.ravel()
        for i in range(len(self.Tdataset)):
            if i > 0:
                plt.subplots_adjust(hspace = 0.4)
            axs[i].plot(self.wavelength, self.NS_Flux_Ratio[i], color = (cMap(colors[i]))-0.1, linewidth=lineWidth, marker = dataPointStyle)
            axs[i].plot([-1000,1000], [1,1], color = 'red', linewidth = lineWidth/2, linestyle = 'dashed')
            axs[i].set_ylabel(yLabel)
            axs[i].set_yticks(yTicks)
            axs[i].set_xlim(xLim)
            axs[i].set_ylim(yLim)
            axs[i].minorticks_on()
            if i == len(self.Tdataset)-1:
                axs[i].set_xticks(xTicks)
                axs[i].xaxis.set_tick_params(labelbottom=True)
            axs[i].yaxis.set_tick_params(labelleft=True)
            axs[i].text(*subplotName[i], (self.Tdataset[i][0] + ' - ' + self.dates[i]), horizontalalignment='center', verticalalignment='center', transform=axs[i].transAxes)
            if i + grid >= len(self.Tdataset):
                axs[i].set_xlabel(xLabel)
        remainder = len(self.Tdataset) % grid
        for x in range(i+1, i+1+remainder):
            fig.delaxes(axs[x])
        fig.suptitle(title)
        #lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        #lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        #fig.legend(fontsize = legendFontSize, loc = legendLocation, markerscale = 2, framealpha = 0, frameon = False, edgecolor = 'white', handlelength = 0)
        plt.show()
        return fig
    def bNS_Flux(self):
        title = "Seasonal Changes in the brightness difference between the North and South of Titan"
        xLabel = "Wavelength (??m)"
        yLabel = "N/S"
        size = [24,24]
        xLim = [0.35,1.04]
        yLim = [0.6,2]
        cmapMin = 0.1
        cmapMax = 1
        cMap = "hsv"
        darken = 0.5
        axisFontSize = 10
        titleFontSize = 10
        legendLocation = 4
        legendFontSize = 9
        lineWidth = 2
        dataPointStyle = ","
        lineStyles = ["solid", "solid", "solid"]
        grid = 1
        subplotName = [[0.035,0.75],[0.035,0.75],[0.035,0.75],[0.035,0.75],[0.035,0.75],[0.035,0.75],[0.035,0.75],[0.035,0.75],[0.035,0.75],[0.038,0.79],[0.038,0.75],[0.038,0.75]]
        self.datasetDates()
        for i in self.allDatasets:
            self.datasetRead(i)
        plt.rcParams["font.family"] = 'times'
        plt.rcParams["font.weight"] = 'light'
        xtickCount = 10
        yTickCount = 4
        xTicks = np.arange(xLim[0], xLim[1]+(xLim[1]-xLim[0])/xtickCount,(xLim[1]-xLim[0])/xtickCount)
        yTicks = np.arange(yLim[0], yLim[1]+(yLim[1]-yLim[0])/yTickCount,(yLim[1]-yLim[0])/yTickCount)
        fig, axs = plt.subplots(nrows = math.ceil(len(self.Tdataset)/grid), ncols = grid, sharex='all', sharey='all', squeeze = False, figsize = size)
        cMap = plt.cm.get_cmap(cMap)
        try:
            colors = np.arange(cmapMin,(cmapMax+((cmapMax-cmapMin)/len(self.Tdataset))),(cmapMax-cmapMin)/len(self.Tdataset))
        except: 
            colors = [cmapMax for i in range(len(self.Tdataset))]
        fig.tight_layout(pad = 2, rect =(0,1.5,1,1))
        fig.subplots_adjust(top=0.95, hspace = 0.8)
        axs = axs.ravel()
        print(cMap(colors[0]))
        for i in range(len(self.Tdataset)):
            if i > 0:
                plt.subplots_adjust(hspace = 0.4)
            axs[i].plot(self.wavelength, self.NS_Flux_Ratio[i], color = ((cMap(colors[i]))[0]*darken,(cMap(colors[i]))[1]*darken,(cMap(colors[i]))[2]*darken,1), linewidth=lineWidth, marker = dataPointStyle, label="Darken")
            axs[i].plot([-1000,1000], [1,1], color = (1,0,0,0.5), linewidth = lineWidth/3, linestyle = 'dashed')
            axs[i].set_ylabel(yLabel)
            axs[i].set_yticks(yTicks)
            axs[i].set_xlim(xLim)
            axs[i].set_ylim(yLim)
            axs[i].minorticks_on()
            if i == len(self.Tdataset)-1:
                axs[i].set_xticks(xTicks)
                axs[i].xaxis.set_tick_params(labelbottom=True)
            axs[i].yaxis.set_tick_params(labelleft=True)
            axs[i].text(*subplotName[i], (self.Tdataset[i][0] + ' - ' + self.dates[i]), horizontalalignment='center', verticalalignment='center', transform=axs[i].transAxes)
            if i + grid >= len(self.Tdataset):
                axs[i].set_xlabel(xLabel)
        remainder = len(self.Tdataset) % grid
        for x in range(i+1, i+1+remainder):
            fig.delaxes(axs[x])
        fig.suptitle(title)
        #lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        #lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        #fig.legend(fontsize = legendFontSize, loc = legendLocation, markerscale = 2, framealpha = 0, frameon = False, edgecolor = 'white', handlelength = 0)
        plt.show()
        return fig
    def cNS_Flux(self):
        title = "Seasonal Changes in the brightness difference between the North and South of Titan"
        xLabel = "Wavelength (??m)"
        yLabel = "N/S"
        size = [12,24]
        xLim = [0.35,1.04]
        yLim = [0.6,2]
        cmapMin = 0.1
        cmapMax = 1
        cMap = "hsv"
        darken = 0.5
        axisFontSize = 12
        titleFontSize = 10
        legendLocation = 4
        legendFontSize = 9
        lineWidth = 2
        ticksize = 15
        datasetSize = 10
        dataPointStyle = ","
        lineStyles = ["solid", "solid", "solid"]
        grid = 1
        subplotName = [[0.075,0.75],[0.075,0.75],[0.075,0.75],[0.075,0.75],[0.075,0.75],[0.075,0.75],[0.075,0.75],[0.075,0.75],[0.075,0.75],[0.075,0.79],[0.075,0.75],[0.075,0.75]]
        self.datasetDates()
        for i in self.allDatasets:
            self.datasetRead(i)
        plt.rcParams["font.family"] = 'times'
        plt.rcParams["font.weight"] = 'light'
        xtickCount = 10
        yTickCount = 4
        xTicks = np.arange(xLim[0], xLim[1]+(xLim[1]-xLim[0])/xtickCount,(xLim[1]-xLim[0])/xtickCount)
        yTicks = np.arange(yLim[0], yLim[1]+(yLim[1]-yLim[0])/yTickCount,(yLim[1]-yLim[0])/yTickCount)
        fig, axs = plt.subplots(nrows = math.ceil(len(self.Tdataset)/grid), ncols = grid, sharex='all', sharey='all', squeeze = False, figsize = size)
        cMap = plt.cm.get_cmap(cMap)
        try:
            colors = np.arange(cmapMin,(cmapMax+((cmapMax-cmapMin)/len(self.Tdataset))),(cmapMax-cmapMin)/len(self.Tdataset))
        except: 
            colors = [cmapMax for i in range(len(self.Tdataset))]
        fig.tight_layout(pad = 2, rect =(0,1.5,1,1))
        fig.subplots_adjust(top=0.95, hspace = 0.8)
        axs = axs.ravel()
        print(cMap(colors[0]))
        fig.text(0.055, 0.5, yLabel, va='center', rotation='vertical', size = axisFontSize)
        for i in range(len(self.Tdataset)):
            if i > 0:
                plt.subplots_adjust(hspace = 0.4)
            axs[i].plot(self.wavelength, self.NS_Flux_Ratio[i], color = ((cMap(colors[i]))[0]*darken,(cMap(colors[i]))[1]*darken,(cMap(colors[i]))[2]*darken,1), linewidth=lineWidth, marker = dataPointStyle, label="Darken")
            axs[i].plot([-1000,1000], [1,1], color = (1,0,0,0.5), linewidth = lineWidth/3, linestyle = 'dashed')
            axs[i].set_yticks(yTicks)
            axs[i].set_xlim(xLim)
            axs[i].set_ylim(yLim)
            axs[i].minorticks_on()
            if i == len(self.Tdataset)-1:
                axs[i].set_xticks(xTicks, size = ticksize)
                axs[i].xaxis.set_tick_params(labelbottom=True)
            axs[i].yaxis.set_tick_params(labelleft=True)
            axs[i].text(*subplotName[i], (self.Tdataset[i][0] + ' - ' + self.dates[i]), horizontalalignment='center', verticalalignment='center', transform=axs[i].transAxes, size = datasetSize)
            if i + grid >= len(self.Tdataset):
                axs[i].set_xlabel(xLabel, size = axisFontSize)
        remainder = len(self.Tdataset) % grid
        for x in range(i+1, i+1+remainder):
            fig.delaxes(axs[x])
        fig.suptitle(title)
        #lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        #lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        #fig.legend(fontsize = legendFontSize, loc = legendLocation, markerscale = 2, framealpha = 0, frameon = False, edgecolor = 'white', handlelength = 0)
        plt.show()
        return fig
    def dNS_Flux(self):
        #prime mission (TA, T8, T31)
        prime = [0,1,2]
        #equinox mission (61,62,67)
        equinox = [3,4,5]
        #solstice mission (79,85,92,108,114) -> (85,108,114)
        solstice = [7,9,10]
        datasets = [prime,equinox,solstice]
        xLabel = "Wavelength (??m)"
        yLabel = "N/S"
        size = [12,24]
        xLim = [0.35,1.04]
        yLim = [0.6,2]
        cmapMin = 0.1
        cmapMax = 1
        cMap = "hsv"
        darken = 0.5
        axisFontSize = 12
        titleFontSize = 10
        legendLocation = 4
        legendFontSize = 9
        lineWidth = 2
        ticksize = 15
        datasetSize = 10
        dataPointStyle = ","
        lineStyles = ["solid", "solid", "solid"]
        grid = 1
        subplotName = [[0.075,0.75],[0.075,0.75],[0.075,0.75],[0.075,0.75],[0.075,0.75],[0.075,0.75],[0.075,0.75],[0.075,0.75],[0.075,0.75],[0.075,0.79],[0.075,0.75],[0.075,0.75]]
        self.datasetDates()
        for i in self.allDatasets:
            self.datasetRead(i)
        plt.rcParams["font.family"] = 'times'
        plt.rcParams["font.weight"] = 'light'
        xtickCount = 10
        yTickCount = 4
        xTicks = np.arange(xLim[0], xLim[1]+(xLim[1]-xLim[0])/xtickCount,(xLim[1]-xLim[0])/xtickCount)
        yTicks = np.arange(yLim[0], yLim[1]+(yLim[1]-yLim[0])/yTickCount,(yLim[1]-yLim[0])/yTickCount)
        fig, axs = plt.subplots(nrows = 3, ncols = grid, sharex='all', sharey='all', squeeze = False, figsize = size)
        cMap = plt.cm.get_cmap(cMap)
        try:
            colors = np.arange(cmapMin,(cmapMax+((cmapMax-cmapMin)/len(self.Tdataset))),(cmapMax-cmapMin)/len(self.Tdataset))
        except: 
            colors = [cmapMax for i in range(len(self.Tdataset))]
        fig.tight_layout(pad = 2, rect =(0,1.5,1,1))
        fig.subplots_adjust(top=0.95, hspace = 0.8)
        axs = axs.ravel()
        print(cMap(colors[0]))
        for i in range(len(datasets)): #iterates over mission
            mission = datasets[i] #mission being iterated over
            fig.text(0.055, 0.5, yLabel, va='center', rotation='vertical', size = axisFontSize)
            plt.subplots_adjust(hspace = 0.4)
            for x in mission:
                flyby = self.Tdataset[x]
                
                axs[i].plot(self.wavelength, self.NS_Flux_Ratio[i], color = ((cMap(colors[i]))[0]*darken,(cMap(colors[i]))[1]*darken,(cMap(colors[i]))[2]*darken,1), linewidth=lineWidth, marker = dataPointStyle, label="Darken")
                axs[i].plot([-1000,1000], [1,1], color = (1,0,0,0.5), linewidth = lineWidth/3, linestyle = 'dashed')
            axs[i].set_yticks(yTicks)
            axs[i].set_xlim(xLim)
            axs[i].set_ylim(yLim)
            axs[i].minorticks_on()
            if i == len(self.Tdataset)-1:
                axs[i].set_xticks(xTicks, size = ticksize)
                axs[i].xaxis.set_tick_params(labelbottom=True)
            axs[i].yaxis.set_tick_params(labelleft=True)
            axs[i].text(*subplotName[i], (self.Tdataset[i][0] + ' - ' + self.dates[i]), horizontalalignment='center', verticalalignment='center', transform=axs[i].transAxes, size = datasetSize)
            if i + grid >= len(self.Tdataset):
                axs[i].set_xlabel(xLabel, size = axisFontSize)
        remainder = len(self.Tdataset) % grid
        #lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        #lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        #fig.legend(fontsize = legendFontSize, loc = legendLocation, markerscale = 2, framealpha = 0, frameon = False, edgecolor = 'white', handlelength = 0)
        plt.show()
    def eNS_Flux(self):
        #prime mission (TA, T8, T31)
        prime = [0,1,2]
        #equinox mission (61,62,67)
        equinox = [3,4,5]
        #solstice mission (79,85,92,108,114) -> (85,108,114)
        solstice = [7,9,10]
        datasets = [prime,equinox,solstice]
        xLabel = "Wavelength (??m)"
        yLabel = "N/S"
        size = [12,24]
        xLim = [0.35,1.05]
        yLim = [0.75,1.5]
        cmapMin = 0.1
        cmapMax = 1
        cMap = "hsv"
        darken = 0.5
        axisFontSize = 12
        titleFontSize = 10
        legendLocation = 4
        legendFontSize = 9
        ticksize = 15
        datasetSize = 10;dataPointStyle = ",";lineStyles = ["solid", "solid", "solid"]; lineWidth = 2
        grid = 1
        subplotName = [[0.075,0.94],[0.075,0.85],[0.075,0.76],[0.075,0.94],[0.075,0.85],[0.075,0.76],[0.075,0.94],[0.075,0.85],[0.075,0.76],[0.075,0.94],[0.075,0.85],[0.075,0.76]]
        self.datasetDates()
        for i in self.allDatasets:
            self.datasetRead(i)
        plt.rcParams["font.family"] = 'times'
        plt.rcParams["font.weight"] = 'light'
        xtickCount = 10;yTickCount = 3;xTicks = np.arange(xLim[0], xLim[1]+(xLim[1]-xLim[0])/xtickCount,(xLim[1]-xLim[0])/xtickCount); yTicks = np.arange(yLim[0], yLim[1]+(yLim[1]-yLim[0])/yTickCount,(yLim[1]-yLim[0])/yTickCount)
        cMap = plt.cm.get_cmap(cMap)
        try:
            colors = np.arange(cmapMin,(cmapMax+((cmapMax-cmapMin)/(len(prime)+len(solstice)+len(equinox)))),(cmapMax-cmapMin)/(len(prime)+len(solstice)+len(equinox)))
        except: 
            colors = [cmapMax for i in range(len(self.Tdataset))]
        fig, axs = plt.subplots(nrows = 3, ncols = grid, sharex='all', sharey='all', squeeze = False, figsize = size)  
        fig.tight_layout(pad = 2, rect =(0,1.5,1,1))
        fig.subplots_adjust(top=0.95, hspace = 0.4)
        axs = axs.ravel()
        print(cMap(colors[0]))
        count = 0
        for i in range(len(datasets)):
            mission = datasets[i]
            axs[i].text(0.055, 0.5, yLabel, va='center', rotation='vertical', size = axisFontSize)
            axs[i].minorticks_on()
            axs[i].set_xlabel(xLabel, size = axisFontSize)            
            axs[i].xaxis.set_tick_params(labelbottom=True)
            axs[i].yaxis.set_tick_params(labelleft=True)  
            for x in mission:
                flyby = self.Tdataset[x][0]
                axs[i].plot(self.wavelength, self.NS_Flux_Ratio[x], color = ((cMap(colors[count]))[0]*darken,(cMap(colors[count]))[1]*darken,(cMap(colors[count]))[2]*darken,1), linewidth=lineWidth, marker = dataPointStyle, label=(self.Tdataset[x][0] + ' - ' + self.dates[x]))
                #xs[i].text(*subplotName[count], (self.Tdataset[x][0] + ' - ' + self.dates[x]), horizontalalignment='center', verticalalignment='center', transform=axs[i].transAxes, size = datasetSize)
                axs[i].set_xlim(xLim)
                axs[i].set_ylim(yLim)
                axs[i].set_yticks(yTicks)
                axs[i].set_xticks(xTicks)
                axs[i].legend()
                count+=1
            axs[i].plot([-1000,1000], [1,1], color = (1,0,0,0.5), linewidth = lineWidth/3, linestyle = 'dashed') 
        #lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        #lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        #fig.legend(fontsize = legendFontSize, loc = legendLocation, markerscale = 2, framealpha = 0, frameon = False, edgecolor = 'white', handlelength = 0)
        plt.show()
        return fig
    def fNS_Flux(self):
        #prime mission (TA, T8, T31)
        prime = [0,1,2]
        #equinox mission (61,62,67)
        equinox = [3,4,5]
        #solstice mission (79,85,92,108,114) -> (85,108,114)
        solstice = [7,9,10]
        datasets = [prime,equinox,solstice]
        xLabel = "Wavelength (??m)"
        yLabel = "N/S Flux Ratio"
        size = [16,24]
        xLim = [0.35,1.05]
        yLim = [0.75,1.5]
        cmapMin = 0.1
        cmapMax = 1
        cMap = "hsv"
        darken = 0.5
        axisFontSize = 12
        titleFontSize = 10
        legendLocation = "lower right"
        legendFontSize = 10
        ticksize = 15
        datasetSize = 10;dataPointStyle = ",";lineStyles = ["solid", "solid", "solid"]; lineWidth = 2
        grid = 1
        subplotName = [[0.075,0.94],[0.075,0.85],[0.075,0.76],[0.075,0.94],[0.075,0.85],[0.075,0.76],[0.075,0.94],[0.075,0.85],[0.075,0.76],[0.075,0.94],[0.075,0.85],[0.075,0.76]]
        self.datasetDates()
        for i in self.allDatasets:
            self.datasetRead(i)
        plt.rcParams["font.family"] = 'times'
        plt.rcParams["font.weight"] = 'light'
        xtickCount = 10;yTickCount = 3;xTicks = np.arange(xLim[0], xLim[1]+(xLim[1]-xLim[0])/xtickCount,(xLim[1]-xLim[0])/xtickCount); yTicks = np.arange(yLim[0], yLim[1]+(yLim[1]-yLim[0])/yTickCount,(yLim[1]-yLim[0])/yTickCount)
        cMap = plt.cm.get_cmap(cMap)
        try:
            colors = np.arange(cmapMin,(cmapMax+((cmapMax-cmapMin)/(len(prime)+len(solstice)+len(equinox)))),(cmapMax-cmapMin)/(len(prime)+len(solstice)+len(equinox)))
        except: 
            colors = [cmapMax for i in range(len(self.Tdataset))]
        fig, axs = plt.subplots(nrows = 3, ncols = grid, sharex='all', sharey='all', squeeze = False, figsize = size)  
        fig.tight_layout(pad = 2, rect =(0,1.5,1,1))
        fig.subplots_adjust(top=0.95, hspace = 0.4)
        axs = axs.ravel()
        print(cMap(colors[0]))
        count = 0; labels = []
        for i in range(len(datasets)):
            mission = datasets[i]
            axs[i].minorticks_on()
            axs[i].set_xlabel(xLabel, size = axisFontSize)               
            axs[i].xaxis.set_tick_params(labelbottom=True)
            axs[i].yaxis.set_tick_params(labelleft=True)  
            for x in mission:
                flyby = self.Tdataset[x][0]
                a = axs[i].plot(self.wavelength, self.NS_Flux_Ratio[x], color = ((cMap(colors[count]))[0]*darken,(cMap(colors[count]))[1]*darken,(cMap(colors[count]))[2]*darken,1), linewidth=lineWidth, marker = dataPointStyle, label=(self.Tdataset[x][0] + ' - ' + self.dates[x]))
                #xs[i].text(*subplotName[count], (self.Tdataset[x][0] + ' - ' + self.dates[x]), horizontalalignment='center', verticalalignment='center', transform=axs[i].transAxes, size = datasetSize)
                axs[i].set_xlim(xLim)
                axs[i].set_ylim(yLim)
                axs[i].set_yticks(yTicks)
                axs[i].set_xticks(xTicks)
                labels.append(a)
                #axs[i].legend()
                count+=1
            axs[i].plot([-1000,1000], [1,1], color = (1,0,0,0.5), linewidth = lineWidth/3, linestyle = 'dashed') 
        #lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        #lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        fig.legend(lines, labels, fontsize = legendFontSize, loc = legendLocation, frameon = True, edgecolor = 'black', bbox_to_anchor = (1,0.2,0,0))
        plt.subplots_adjust(right = .875)
        fig.text(0.055, 0.5, yLabel, va='center', rotation='vertical', size = axisFontSize*1.25)
        plt.show()
        return fig
    def gNS_Flux(self):
        #prime mission (TA, T8, T31)
        prime = [0,1,2]
        #equinox mission (61,62,67, 79,85)
        equinox = [3,5,7]
        #solstice mission (92,108,114,278,283) -> (85,108,114)
        solstice = [9,11,12]
        datasets = [prime,equinox,solstice]
        xLabel = "Wavelength (??m)"
        yLabel = "N/S Flux Ratio"
        size = [16,24]
        xLim = [0.35,1.05]
        xTicks = [0.35,0.45,0.55,0.65,0.75,0.85,0.95,1.05]
        yLim = [0.75,1.5]
        Seasons = ["Northern Winter", " Northern Spring", "Northern Summer"]
        cmapMin = 0.1
        cmapMax = 1
        cMap = "hsv"
        darken = 0.5
        axisFontSize = 20
        titleFontSize = 20
        legendLocation = "lower right"
        legendFontSize = 15
        ticksize = 16
        datasetSize = 10;dataPointStyle = ",";lineStyles = ["solid", "solid", "solid"]; lineWidth = 2
        grid = 1
        subplotName = [[0.075,0.94],[0.075,0.85],[0.075,0.76],[0.075,0.94],[0.075,0.85],[0.075,0.76],[0.075,0.94],[0.075,0.85],[0.075,0.76],[0.075,0.94],[0.075,0.85],[0.075,0.76]]
        self.datasetDates()
        for i in self.allDatasets:
            self.datasetRead(i)
        plt.rcParams["font.family"] = 'times'
        plt.rcParams["font.weight"] = 'light'
        yTickCount = 3; yTicks = np.arange(yLim[0], yLim[1]+(yLim[1]-yLim[0])/yTickCount,(yLim[1]-yLim[0])/yTickCount)
        cMap = plt.cm.get_cmap(cMap)
        try:
            colors = np.arange(cmapMin,(cmapMax+((cmapMax-cmapMin)/(len(prime)+len(solstice)+len(equinox)))),(cmapMax-cmapMin)/(len(prime)+len(solstice)+len(equinox)))
        except: 
            colors = [cmapMax for i in range(len(self.Tdataset))]
        fig, axs = plt.subplots(nrows = 3, ncols = grid, sharex='all', sharey='all', squeeze = False, figsize = size)  
        fig.tight_layout(pad = 2, rect =(0,1.5,1,1))
        fig.subplots_adjust(top=.85, hspace = 0.2, left = 0.12, right = 0.2)
        axs = axs.ravel()
        print(cMap(colors[0]))
        count = 0; labels = []
        for i in range(len(datasets)):
            mission = datasets[i]
            axs[i].text(.36,1.35, Seasons[i], fontsize = 20)
            axs[i].minorticks_on()
            if i == 0:             
                axs[i].xaxis.set_tick_params(labelbottom=False, labeltop=True, bottom =False, top = True, which = 'both')
                axs[i].xaxis.set_label_position('top')
                # axs[i].set_xlabel(xLabel, size = axisFontSize)
                axs[i].minorticks_on()
            elif i == 2:
                axs[i].xaxis.set_tick_params(labelbottom=True, labeltop =False)
                axs[i].xaxis.set_label_position('bottom') 
                axs[i].set_xlabel(xLabel, size = axisFontSize)
            else:
                axs[i].xaxis.set_tick_params(labelbottom=False, labeltop=False)
            axs[i].yaxis.set_tick_params(labelleft=True)
            axs[i].set_yticks(ticks = np.round(np.arange(yLim[0],yLim[1]+ (yLim[1]-yLim[0])/5,(yLim[1]-yLim[0])/5),4))
            axs[i].set_yticklabels(labels = np.round(np.arange(yLim[0],yLim[1]+ (yLim[1]-yLim[0])/5,(yLim[1]-yLim[0])/5),4),fontsize = ticksize)
            axs[i].set_xticks(ticks = xTicks)
            axs[i].set_xticklabels(labels = xTicks, fontsize = ticksize)
            for x in mission:
                flyby = self.Tdataset[x][0]
                a = axs[i].plot(self.wavelength, self.NS_Flux_Ratio[x], color = ((cMap(colors[count]))[0]*darken,(cMap(colors[count]))[1]*darken,(cMap(colors[count]))[2]*darken,1), linewidth=lineWidth, marker = dataPointStyle, label=(self.Tdataset[x][0] + ' - ' + self.dates[x]))
                #xs[i].text(*subplotName[count], (self.Tdataset[x][0] + ' - ' + self.dates[x]), horizontalalignment='center', verticalalignment='center', transform=axs[i].transAxes, size = datasetSize)
                axs[i].set_xlim(xLim)
                axs[i].set_ylim(yLim)
                labels.append(a)
                #axs[i].legend()
                count+=1
            axs[i].plot([-1000,1000], [1,1], color = (1,0,0,0.5), linewidth = lineWidth, linestyle = 'dashed') 
        #lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        #lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        plt.subplots_adjust(right = .8, hspace = 0.12)
        fig.legend(lines, labels, fontsize = legendFontSize, loc = legendLocation, frameon = False, edgecolor = 'black', bbox_to_anchor = (1,0.2,0,0))
        
        fig.text(0.055, 0.5, yLabel, va='center', rotation='vertical', size = axisFontSize*1.25)
        plt.show()
        return fig           
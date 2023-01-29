import os
import os.path as path
from tkinter import font
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as plter
import math
class Boundary:
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
            self.allDatasets.append(os.path.join(self.directory["flyby_parent_directory"], self.directory["analysis_folder"], self.i[0] + self.directory["analysis"]).replace("\\","/"))

    def createFigureFolder(self):
        folderPath = os.path.join(self.directory["flyby_parent_directory"], self.directory["Figure names"]["figures"]).replace("\\","/")
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
        self.resultsFolder = folderPath
    def createFileFolder(self):
        folderPath = os.path.join(self.directory["flyby_parent_directory"], self.directory["Figure names"]["if"]).replace("\\","/")
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
        self.resultsFolder = folderPath
    def wavelengths(self):
        self.wavelength = (np.array(pd.read_csv(os.path.join(self.directory["flyby_parent_directory"], self.directory["wavelength_data"]).replace("\\","/"), header = None)))[0]
    def datasetDates(self):
        self.dates = []
        date = np.array(pd.read_csv(os.path.join(self.directory["flyby_parent_directory"], self.directory["flyby_info"]).replace("\\","/"), header = None))
        for i in self.Tdataset:
            rowOne = date[:, 0]
            rowOne = rowOne.tolist()
            row = rowOne.index(i[0])
            self.dates.append(date[row,2])
    def datasetRead(self, x):
        self.data = np.array(pd.read_csv(x, header = None))
        self.NSA.append(((self.data[0])[1:-1]).astype(np.float64))
        self.NS_Flux_Ratio.append(((self.data[2])[1:-1]).astype(np.float64))
    def a_boundary(self):
        #prime mission (TA, T8, T31)
        prime = [0,1,2]
        #equinox mission (61,62,67, 79,85)
        equinox = [3,5,7]
        #solstice mission (92,108,114,278,283) -> (85,108,114)
        solstice = [9,11,12]
        datasets = [prime,equinox,solstice]
        xLabel = "Wavelength (µm)"
        yLabel = "NSA Latitude (°)"
        size = [16,24]
        xLim = [0.35,1.05]
        xTicks = [0.35,0.45,0.55,0.65,0.75,0.85,0.95,1.05]
        # yTicks = []
        yLim = [-15,15]
        yText = [10,10,-12]
        yTicks = [-15, -10,-5,0,5,10,15]
        Seasons = ["Northern Winter", " Northern Spring", "Northern Summer"]
        cmapMin = 0
        cmapMax = 1
        darken = 1.0
        colors = [plt.cm.get_cmap("autumn"),plt.cm.get_cmap("cool"),plt.cm.get_cmap("winter")]
        color = []
        for col in colors:
            col = [[c*darken for index, c in enumerate(col(index/2)) if index != 3] for index in range(3)]
            color.extend(col)
        colors = color
        colors = [
            (100,0,150,1),
            (245,0,48,1),
            (0,190,212,1),
            (50,0,130,1),
            (0,130,200,1),
            (100,190,150,1),
            (150,150,150,1),
            (154,181,185,1),
            (0,0,0,1),
        ]
        for index, col in enumerate(colors):
            colors[index] = [c/255 for ind, c in enumerate(col) if ind !=3]
            
        cMap = plt.cm.get_cmap("tab10")
        colors = np.arange(0,1.2,1/9)
        axisFontSize = 20
        titleFontSize = 20
        legendLocation = "center left"
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
        fig, axs = plt.subplots(nrows = 3, ncols = grid, sharex='all', sharey='all', squeeze = False, figsize = size)  
        fig.tight_layout(pad = 2, rect =(0,1.5,1,1))
        fig.subplots_adjust(top=.85, hspace = 0.2, left = 0.12, right = 0.2)
        axs = axs.ravel()
        count = 0; labels = []
        for i in range(len(datasets)):
            mission = datasets[i]
            axs[i].text(.36,yText[i], Seasons[i], fontsize = 20)
            axs[i].minorticks_on()
            if i == 0:             
                axs[i].xaxis.set_tick_params(labelbottom=False, labeltop=True, bottom =False, top = True, which = 'both')
                axs[i].xaxis.set_label_position('top')
                # axs[i].set_xlabel(xLabel, size = axisFontSize)
                axs[i].minorticks_on()
            elif i == 2:
                axs[i].xaxis.set_tick_params(labelbottom=True, labeltop =False)
                axs[i].xaxis.set_label_position('bottom') 
                axs[i].set_xlabel(xLabel, size = axisFontSize*1.25)
            else:
                axs[i].xaxis.set_tick_params(labelbottom=False, labeltop=False)
            axs[i].yaxis.set_tick_params(labelleft=True)
            axs[i].set_yticks(ticks = yTicks)
            axs[i].set_yticklabels(labels = yTicks,fontsize = ticksize)
            axs[i].set_xticks(ticks = xTicks)
            axs[i].set_xticklabels(labels = xTicks, fontsize = ticksize)
            for x in mission:
                flyby = self.Tdataset[x][0]
                if count == 2:
                    a = axs[i].plot(self.wavelength, self.NSA[x], color = (0.1,0.8,0.2,1), linewidth=lineWidth, marker = dataPointStyle, label=(self.Tdataset[x][0] + ' - ' + self.dates[x]))
                elif count == 1:
                    a = axs[i].plot(self.wavelength, self.NSA[x], color = ((cMap(colors[count]))[0]*0.5,(cMap(colors[count]))[1]*0.5,(cMap(colors[count]))[2]*0.5,1), linewidth=lineWidth, marker = dataPointStyle, label=(self.Tdataset[x][0] + ' - ' + self.dates[x]))
                else:
                    a = axs[i].plot(self.wavelength, self.NSA[x], color = ((cMap(colors[count]))[0]*darken,(cMap(colors[count]))[1]*darken,(cMap(colors[count]))[2]*darken,1), linewidth=lineWidth, marker = dataPointStyle, label=(self.Tdataset[x][0] + ' - ' + self.dates[x]))
                # a = axs[i].plot(self.wavelength, self.NSA[x], color = colors[count], linewidth=lineWidth, marker = dataPointStyle, label=(self.Tdataset[x][0] + ' - ' + self.dates[x]))
                #xs[i].text(*subplotName[count], (self.Tdataset[x][0] + ' - ' + self.dates[x]), horizontalalignment='center', verticalalignment='center', transform=axs[i].transAxes, size = datasetSize)
                axs[i].set_xlim(xLim)
                axs[i].set_ylim(yLim)
                labels.append(a)
                #axs[i].legend()
                count+=1
            lines_labels = axs[i].get_legend_handles_labels()
            lines, labels = lines_labels
            axs[i].plot([-1000,1000], [0,0], color = (1,0,0,0.8), linewidth = lineWidth, linestyle = 'dashed') 
            axs[i].legend(lines, labels, fontsize = legendFontSize, loc = legendLocation, frameon = False, edgecolor = 'black', bbox_to_anchor = (1.0,0.25,0.6,0.4))
        #lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        #lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        # lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        # lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        plt.subplots_adjust(right = .8, hspace = 0.12)
        # fig.legend(lines, labels, fontsize = legendFontSize, loc = legendLocation, frameon = False, edgecolor = 'black', bbox_to_anchor = (1,0.2,0,0))
        
        fig.text(0.055, 0.5, yLabel, va='center', rotation='vertical', size = axisFontSize*1.25)
        plt.show()
        return fig           

import os
import os.path as path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from data import data
class if_sh_Figure:
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
    def createFigureFolder(self):
        folderPath = self.directory[0] + "/" + self.directory[8][0]
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
        self.resultsFolder = folderPath
    def createFileFolder(self):
        folderPath = self.directory[0] + "/" + self.directory[8][0] + "/" + self.directory[8][2]
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
    def aIF(self, title, xLabel, yLabel, bands, size, cMap, axisFontSize, titleFontSize,legendLocation, legendFontSize, lineWidth, dataPointStyle, lineStyles, grid, cmapMin = 0, cmapMax = 1):
        self.wavelengths()
        self.datasetDates()
        purpose = ["if_sh",bands, ""]
        plt.rcParams["font.family"] = 'monospace'
        plt.rcParams["font.weight"] = 'light'
        xTicks = range(-50, 51, 10)
        yTicks = np.arange(-0.1, 0.125, 0.025)
        cMap = plt.cm.get_cmap(cMap)
        colors = np.arange(cmapMin,(cmapMax+((cmapMax-cmapMin)/len(bands))),(cmapMax-cmapMin)/len(bands))
        fig, axs = plt.subplots(nrows = math.ceil(len(self.Tdataset)/grid), ncols = grid, sharex='all', sharey='all', squeeze = False, figsize = size)
        plt.subplots_adjust(hspace = 0.4)
        axs = axs.ravel()
        for i in range(len(self.Tdataset)):
            currentDataset = data(self.directory,self.Tdataset[i], self.shiftDegree, purpose)
            for band in range(len(bands)):
                x = currentDataset.if_sh[band][0]
                y = currentDataset.if_sh[band][1]
                if i == 0:
                    axs[i].plot(x,y, lw = lineWidth, linestyle = lineStyles[band], color = cMap(colors[band]), label = str(self.wavelength[band]) +"µm" )
                else:
                    axs[i].plot(x,y, lw = lineWidth, linestyle = lineStyles[band], color = cMap(colors[band]))
            axs[i].set_ylabel(yLabel)
            setTick = max(abs(x))
            axs[i].set_xticks(xTicks)
            axs[i].set_yticks(yTicks)
            axs[i].set_xlim(min(xTicks),max(xTicks))
            axs[i].set_ylim(min(yTicks),max(yTicks))
            axs[i].minorticks_on()
            axs[i].xaxis.set_tick_params(labelbottom=True)
            axs[i].yaxis.set_tick_params(labelleft=True)
            axs[i].text(0.94, 0.8, self.Tdataset[i][0], horizontalalignment='center', verticalalignment='center', transform=axs[i].transAxes)
            if i + grid >= len(self.Tdataset):
                axs[i].set_xlabel(xLabel)
        remainder = len(self.Tdataset) % grid
        for x in range(i+1, i+1+remainder):
            fig.delaxes(axs[x])
        fig.suptitle(title)
        lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        fig.legend(lines, labels, loc = legendLocation)
        plt.show()
    def bIF(self, title, xLabel, yLabel, bands, size, cMap, axisFontSize, titleFontSize,legendLocation, legendFontSize, lineWidth, dataPointStyle,lineStyles, grid, cmapMin = 0, cmapMax = 1):
        self.wavelengths()
        self.datasetDates()
        purpose = ["if_sh",bands, ""]
        plt.rcParams["font.family"] = 'monospace'
        plt.rcParams["font.weight"] = 'light'
        grid = 2
        xTicks = range(-90, 91, 10)
        yTicks = np.arange(-30, 31, 15)
        cMap = plt.cm.get_cmap(cMap)
        colors = np.arange(cmapMin,(cmapMax+((cmapMax-cmapMin)/len(self.Tdataset))),(cmapMax-cmapMin)/len(bands))
        plt.figure(size)
        fig, axs = plt.subplots(nrows = math.ceil(len(self.Tdataset)/grid), ncols = grid, sharex='all', sharey='all', squeeze = False, figsize = size)
        plt.subplots_adjust(hspace = 0.4)
        axs = axs.ravel()
        for i in range(len(self.Tdataset)):
            currentDataset = data(self.directory,self.Tdataset[i], self.shiftDegree, purpose)
            for band in range(len(bands)):
                x = currentDataset.if_sh[band][0]
                y = currentDataset.if_sh[band][1]
                if i == 0:
                    axs[i].plot(x,y, lw = lineWidth, linestyle = lineStyles[band], color = cMap(colors[band]), label = str(self.wavelength[band]) +"µm" )
                else:
                    axs[i].plot(x,y, lw = lineWidth, linestyle = lineStyles[band], color = cMap(colors[band]))
            axs[i].set_ylabel(yLabel)
            setTick = max(abs(x))
            axs[i].set_xticks(xTicks)
            axs[i].set_yticks(yTicks)
            axs[i].set_xlim(min(xTicks),max(xTicks))
            axs[i].set_ylim(min(yTicks),max(yTicks))
            axs[i].minorticks_on()
            axs[i].xaxis.set_tick_params(labelbottom=True)
            axs[i].yaxis.set_tick_params(labelleft=True)
            axs[i].title.set_text(self.Tdataset[i][0])
            if i + grid >= len(self.Tdataset):
                axs[i].set_xlabel(xLabel)
        remainder = len(self.Tdataset) % grid
        for x in range(i+1, i+1+remainder):
            fig.delaxes(axs[x])
        fig.suptitle(title)
        lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        fig.legend(lines, labels, loc = legendLocation)
        plt.show()
    def cIF(self):
        title = "Seasonal Evolution of Titan's Atmospheric Meridional Brightness Profile"
        xLabel = "Latitude"
        yLabel = "I/F"
        bands = [0, 47, 95]
        size = [16,39]
        cMap = "viridis"
        axisFontSize = 9
        tickFontSize = 4
        titleFontSize = 15
        legendFontSize = 6
        lineWidth = 1
        dataPointStyle = "."
        lineStyles = ["solid", "solid", "solid"]
        grid = 1
        cmapMin = 0
        cmapMax = 1
        self.wavelengths()
        self.datasetDates()
        purpose = ["if_sh",bands, ""]
        plt.rcParams["font.family"] = 'monospace'
        plt.rcParams["font.weight"] = 'light'
        xTicks = range(-50, 51, 10)
        yTicks = np.arange(-0.125, 0.1875, 0.0625)
        cMap = plt.cm.get_cmap(cMap)
        colors = np.arange(cmapMin,(cmapMax+((cmapMax-cmapMin)/len(bands))),(cmapMax-cmapMin)/len(bands))
        fig, axs = plt.subplots(nrows = math.ceil(len(self.Tdataset)/grid), ncols = grid, sharex='all', sharey='all', squeeze = False, figsize = size)
        axs = axs.ravel()
        for i in range(len(self.Tdataset)):
            currentDataset = data(self.directory,self.Tdataset[i], self.shiftDegree, purpose)
            for band in range(len(bands)):
                x = currentDataset.if_sh[band][0]
                y = currentDataset.if_sh[band][1]
                if i == 0:
                    axs[i].plot(x,y, lw = lineWidth, linestyle = lineStyles[band], color = cMap(colors[band]), label = str(self.wavelength[bands[band]]) +"µm" )
                else:
                    axs[i].plot(x,y, lw = lineWidth, linestyle = lineStyles[band], color = cMap(colors[band]))
            axs[i].plot((-100,100),(0,0), lw = 0.5, color = (1,0,0))
            axs[i].set_ylabel(yLabel)
            setTick = max(abs(x))
            axs[i].set_xlim(min(xTicks),max(xTicks))
            axs[i].set_ylim(min(yTicks),max(yTicks))
            axs[i].minorticks_on()
            axs[i].xaxis.set_tick_params(labelbottom=True, labelsize = axisFontSize)
            axs[i].set_yticks(yTicks)
            axs[i].yaxis.set_tick_params(labelleft=True, labelsize = axisFontSize)
            axs[i].set_xticks(xTicks, fontsize= tickFontSize)

            axs[i].text(0.98, 0.75, self.Tdataset[i][0], horizontalalignment='center', verticalalignment='center', transform=axs[i].transAxes)
        remainder = len(self.Tdataset) % grid
        axs[i].set_xlabel(xLabel)
        fig.suptitle(title)
        lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        fig.legend(lines, labels, loc = 'lower right', frameon = False, bbox_to_anchor = (-0.01,0.15,1,1))
        fig.tight_layout(pad = 1.2) #,rect =(0,1.5,1,1)
        fig.subplots_adjust(top = 0.95, right = 0.9, bottom = 0.05, left = 0.08, hspace = .6)
        for i in axs:
            i.height =  size[1]
        plt.show()
        return fig
    def dIF(self):
        title = "Seasonal Evolution of Titan's Atmospheric Meridional Brightness Profile"
        xLabel = "Latitude"
        yLabel = "I/F"
        bands = [0, 47, 95]
        size = [16,32]
        cMap = "viridis"
        axisFontSize = 9
        label = 12
        tickFontSize = 4
        titleFontSize = 15
        legendFontSize = 6
        lineWidth = 1
        dataPointStyle = "."
        lineStyles = ["solid", "solid", "solid"]
        grid = 2
        cmapMin = 0
        cmapMax = 1
        self.wavelengths()
        self.datasetDates()
        purpose = ["if_sh",bands, ""]
        plt.rcParams["font.family"] = 'monospace'
        plt.rcParams["font.weight"] = 'light'
        xTicks = range(-50, 51, 10)
        yTicks = np.arange(-0.125, 0.1875, 0.0625)
        cMap = plt.cm.get_cmap(cMap)
        colors = np.arange(cmapMin,(cmapMax+((cmapMax-cmapMin)/len(bands))),(cmapMax-cmapMin)/len(bands))
        fig, axs = plt.subplots(nrows = math.ceil(len(self.Tdataset)/grid), ncols = grid, sharex='all', sharey='all', squeeze = False, figsize = size)
        axs = axs.ravel()
        for i in range(len(self.Tdataset)):
            currentDataset = data(self.directory,self.Tdataset[i], self.shiftDegree, purpose)
            for band in range(len(bands)):
                x = currentDataset.if_sh[band][0]
                y = currentDataset.if_sh[band][1]
                if i == 0:
                    axs[i].plot(x,y, lw = lineWidth, linestyle = lineStyles[band], color = cMap(colors[band]), label = str(self.wavelength[bands[band]]) +"µm" )
                else:
                    axs[i].plot(x,y, lw = lineWidth, linestyle = lineStyles[band], color = cMap(colors[band]))
            axs[i].plot((-100,100),(0,0), lw = 0.5, color = (1,0,0), linestyle = '--')
            #axs[i].set_ylabel(yLabel)
            setTick = max(abs(x))
            axs[i].set_xlim(min(xTicks),max(xTicks))
            axs[i].set_ylim(min(yTicks),max(yTicks))
            axs[i].minorticks_on()
            axs[i].xaxis.set_tick_params(labelbottom=True, labelsize = axisFontSize)
            axs[i].set_yticks(yTicks)
            axs[i].yaxis.set_tick_params(labelleft=True, labelsize = axisFontSize)
            axs[i].set_xticks(xTicks, fontsize= tickFontSize)

            axs[i].text(0.95, 0.8, self.Tdataset[i][0], horizontalalignment='center', verticalalignment='center', transform=axs[i].transAxes)
        fig.text(0.49, 0.03, xLabel, size = label, horizontalalignment='center', verticalalignment='center')
        fig.text(0.03, 0.52, yLabel, size = label, horizontalalignment='center', verticalalignment='center', rotation = 'vertical')
        fig.suptitle(title)
        lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        fig.legend(lines, labels, loc = 'lower right', frameon = False, bbox_to_anchor = (-0.01,0.15,1,1))
        #fig.tight_layout(pad = 1.2) #,rect =(0,1.5,1,1)
        fig.subplots_adjust(top = 0.95, right = 0.9, bottom = 0.1, left = 0.08, hspace = .75)
        for i in axs:
            i.height =  size[1]
        plt.show()
        return fig
    def eIF(self):
        #prime mission (TA, T8, T31)
        prime = [0,1,2]
        #equinox mission (61,62,67)
        equinox = [3,4,5]
        #solstice mission (79,85,92,108,114) -> (85,108,114)
        solstice = [7,9,10]
        datasets = [prime,equinox,solstice]

        xLabel = "Latitude (°)"
        yLabel = "I/F"
        bands = [0, 47, 95]
        size = [16,32]
        cMap = "Greys"
        axisFontSize = 9
        label = 12
        tickFontSize = 4
        titleFontSize = 15
        legendFontSize = 6
        lineWidth = 1.5
        dataPointStyle = "."
        lineStyles = ["solid", (0, (1, 1)), (0, (5, 1))]
        grid = 2
        cmapMin = 0.5
        cmapMax = 1
        self.wavelengths()
        self.datasetDates()
        purpose = ["if_sh",bands, ""]
        plt.rcParams["font.family"] = 'monospace'
        plt.rcParams["font.weight"] = 'light'
        xTicks = range(-50, 51, 10)
        yTicks = np.arange(-0.075, 0.125, 0.025)
        cMap = plt.cm.get_cmap(cMap)
        numColors = (len(solstice))
        colors = np.arange(cmapMin,(cmapMax+((cmapMax-cmapMin)/numColors)),(cmapMax-cmapMin)/numColors)
        figures = []
        for mission in datasets:
            cColor = 0
            fig = plt.figure(figsize = size)
            plt.xlabel(xLabel, size = label)
            plt.ylabel(yLabel, size = label)
            plt.xlim(min(xTicks),max(xTicks))
            plt.ylim(min(yTicks),max(yTicks))
            for d in range(len(mission)):
                dataset = mission[d]
                currentDataset = data(self.directory,self.Tdataset[dataset], self.shiftDegree, purpose)
                for band in range(len(bands)):
                    x = currentDataset.if_sh[band][0]
                    y = currentDataset.if_sh[band][1]
                    plt.plot(x,y, lw = lineWidth, linestyle = lineStyles[band], color = cMap(colors[cColor]), label = self.Tdataset[dataset][0] + " at " + str(self.wavelength[bands[band]]) +"µm" )
                    plt.plot((-100,100),(0,0), lw = 0.5, color = (1,0,0), linestyle = '--')
                cColor+=1
            plt.legend()
            figures.append(fig)
            plt.show()
        return figures
    def IFScale(self):
        self.ifScale = []
        scales = [["Ta", [725,361]],["T8", [725,361]], ["T31", [725,361]], ["T61", [725,361]], ["T62", [133,125]], ["T67", [725,361]], ["T79", [725,361]],["T85",[725,361]], ["T92", [725,361]], ["T108", [725,361]], ["T114", [363,181]], ["278TI",[724,361]], ["283TI",[724,361]]]
        ifs = np.array(pd.read_csv(self.directory[0] + '/' + self.directory[1] + '/' + self.directory[10], header = None), dtype=object)
        for i in range(len(ifs)):
            self.ifScale.append((scales[i][0], list(ifs[i][:])))
    def fIF(self):
        self.IFScale()
        #prime mission (TA, T8, T31)
        prime = [0,1,2]
        #equinox mission (61,62,67)
        equinox = [3,4,5]
        #solstice mission (79,85,92,108,114) -> (85,108,114)
        solstice = [7,9,10]
        datasets = [prime,equinox,solstice]
        yLabel = "Latitude (°)"
        xLabel = "I/F"
        bands = [0, 47, 95]
        size = [16,32]
        cMap = "Greys"
        axisFontSize = 9
        label = 16
        tickFontSize = 4
        titleFontSize = 15
        legendFontSize = 6
        lineWidth = 1.5
        dataPointStyle = "."
        lineStyles = ["solid", (0, (1, 1)), (0, (5, 1))]
        grid = 2
        cmapMin = 0.5
        cmapMax = 1
        self.wavelengths()
        self.datasetDates()
        purpose = ["if_sh",bands, ""]
        plt.rcParams["font.family"] = 'monospace'
        plt.rcParams["font.weight"] = 'light'
        yTicks = range(-90, 91, 10)
        xTicks = np.arange(0,0.08, 0.005)
        cMap = plt.cm.get_cmap(cMap)
        numColors = (len(solstice))
        colors = [(.75,0.1,0,1), (0,.75,0,1), (0,0.5,1,1)]
        figures = []
        yTick = [str(i) + "°N" if i >= 0 else str(abs(i)) + "°S" for i in list(yTicks)]
        try:
            try:
                a = yTick.index("0.0°N")
            except:
                a = yTick.index("0°N")
            yTick[a] = "0°"
        except:
            pass
        for mission in datasets:
            cColor = 0
            fig = plt.figure(figsize = size)
            plt.xlabel(xLabel, size = label)
            plt.ylabel(yLabel, size = label)
            plt.xticks(xTicks,fontsize=14)
            plt.yticks(ticks = yTicks, labels = yTick,fontsize=14)
            plt.xticks(xTicks)
            scales= [i[0] for i in self.ifScale]
            for d in range(len(mission)):
                dataset = mission[d]
                currentDataset = data(self.directory,self.Tdataset[dataset], self.shiftDegree, purpose)
                for band in range(len(bands)):
                    y = currentDataset.if_sh[band][0]
                    x = currentDataset.if_sh[band][1]
                    if self.Tdataset[dataset][0] in scales :
                        cData = scales.index(self.Tdataset[dataset][0])
                        if dataset == 11:
                            x /=self.ifScale[cData][1][band]
                        else:
                            x /=self.ifScale[cData][1][band]
                        print(band,self.Tdataset[dataset][0] )
                    else:
                        print(self.Tdataset[dataset][0])
                    plt.plot(x,y, lw = lineWidth, linestyle = lineStyles[band], color = colors[d], label = self.Tdataset[dataset][0] + " at " + str(self.wavelength[bands[band]]) +"µm" )
                    #plt.plot((0,0),(-100,100), lw = 1, color = (0,0,0), linestyle = '--')
                cColor+=1
            plt.legend(fontsize = "large")
            figures.append(fig)
            plt.show()
        return figures
    def gIF(self):
        self.IFScale()
        #prime mission (TA, T8, T31)
        prime = [0,1,2]
        #equinox mission (61,62,67, 79,85)
        equinox = [3,5,7]
        #solstice mission (92,108,114,278,283) -> (85,108,114)
        solstice = [9,11,12]
        datasets = [prime,equinox,solstice]
        yLabel = "Latitude"
        xLabel = "I/F"
        bands = [27, 89]
        size = [16  ,32];
        cMap = "gist_ncar"
        axisFontSize = 9
        label = 24
        tickFontSize = 4
        titleFontSize = 20
        legendFontSize = 10
        lineWidth = 4
        dataPointStyle = "."
        lineStyles = ["solid", "solid","solid"]
        grid = 2
        cmapMin = 0.2
        cmapMax = .8
        self.wavelengths()
        self.datasetDates()
        purpose = ["if_sh",bands, ""]
        plt.rcParams["font.family"] = 'monospace'
        plt.rcParams["font.weight"] = 'light'
        yTicks = range(-90, 91, 30)
        xTicks = np.arange(0,0.08, 0.015)
        cMap = plt.cm.get_cmap(cMap)
        numColors = (len(solstice))
        colors = [(.75,0.1,0,1), (0,.75,0,1), (0,0.5,1,1)]
        # try:
        #     colors = np.arange(cmapMin,(cmapMax+((cmapMax-cmapMin)/(9))),(cmapMax-cmapMin)/(9))
        # except:
        #     colors = [cmapMax for i in range(len(self.Tdataset))]
        # colors = np.flip(colors)
        figures = []
        yTick = [str(i) + "°N" if i >= 0 else str(abs(i)) + "°S" for i in list(yTicks)]
        try:
            try:
                a = yTick.index("0.0°N")
            except:
                a = yTick.index("0°N")
            yTick[a] = "0°"
        except:
            pass
        for band in range(len(bands)):
            cColor = 0
            fig = plt.figure(figsize = size)
            plt.xlabel(xLabel, size = label)
            plt.ylabel(yLabel, size = label)
            plt.xlim(min(xTicks),max(xTicks))
            plt.ylim(min(yTicks),max(yTicks))
            plt.xticks(xTicks,fontsize=18)
            plt.yticks(ticks = yTicks, labels = yTick,fontsize=18)
            for mission in datasets:
                scales= [i[0] for i in self.ifScale]
                for d in range(len(mission)):
                    dataset = mission[d]
                    currentDataset = data(self.directory,self.Tdataset[dataset], self.shiftDegree, purpose)
                    y = currentDataset.if_sh[band][0]
                    x = currentDataset.if_sh[band][1]
                    if self.Tdataset[dataset][0] in scales:
                        cData = scales.index(self.Tdataset[dataset][0])
                        if dataset == 7:
                            x = x[14::]
                            y = y[14::]
                            x /=self.ifScale[cData][1][band]
                        if dataset == 11:
                            x *=self.ifScale[cData][1][band]/255
                            x = x[0:240]
                            y = y[0:240]
                        elif dataset == 12:
                            x *=self.ifScale[cData-1][1][band]/255
                            x = x[0:211]
                            y = y[0:211]
                            print(max(x), min(x))
                        else:
                            x /=self.ifScale[cData][1][band]
                        print(band,self.Tdataset[dataset][0] )
                    else:
                        print(self.Tdataset[dataset][0])
                    plt.plot(x,y, lw = lineWidth, linestyle = lineStyles[band], color = [cMap(cColor/len(scales))[0]*0.5,cMap(cColor/len(scales))[1]*0.7,cMap(cColor/len(scales))[2]*0.7,1], label = self.Tdataset[dataset][0], solid_capstyle="butt")
                    #plt.plot((0,0),(-100,100), lw = 1, color = (0,0,0), linestyle = '--')
                    # if min(y) > -89:
                    #     plt.plot((x[-1]-0.0005, x[-1]+0.0005),(y[-1],y[-1]), color = (0,0,0,1), lw = 2)
                    # if max(y) < 89:
                    #     plt.plot((x[0]-0.00005, x[0]+0.0005),(y[0],y[0]), color = (0,0,0,1), lw = 2)
                    cColor+=1
            plt.legend(fontsize = "x-large", frameon = False)
            figures.append(fig)
            plt.figtext(0.5,0.9, str(self.wavelength[bands[band]]) +"µm" , fontsize = 20)
            plt.show()
        return figures
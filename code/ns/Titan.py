from comparison import ComparisonToRoman
from data import data
from flux import NS_Flux_Ratio
from if_sh_figure import if_sh_Figure
from print import printDatasets
from tif import tif
from tilt import Tilt
import numpy as np
import time
import subprocess, os, platform
class Titan:
    def __init__(self, directory = ['C:/Users/aadvi/Desktop/North_South_Asymmetry/data', 'Titan Data', 'csv', 'vis.cyl', 'wavelength.csv','_analytics.csv', 'nsa_cubes_northernsummer.csv', 'Result', ['Figures', 'ComparisonToRoman', 'IF Subplots', 'NS Flux Ratio', 'Tilt', 'Flowchart'], 'roman.csv', 'IFData.csv'], shiftDegree = 6, datasets =  [['Ta', -12, 15, [110,290]],['T8', -12, 15, [120,300]],['T31', -12, 15, [60,240]],['T61', -12, 15,[-120,-300]],['T62', -12, 15,  [79,247]],['T67', -12, 15, [-120,-300]],['T79', -10, 15, [60,240]],['T85', -10, 15, [60,240]],['T92', -5, 15, [100,280]],['T108', -0, 15, [100,280]],['T114', 0, 20, [60,240]],['278TI', 10, 25, [60,240]],['283TI', 10, 25, [60,240]]], purpose = ["if_sh", [71,72,73], "show"], whichDatasets = True,  info = []):
        self.directory = directory
        self.datasets = [dataset[0] for dataset in datasets]
        self.datasetsNSA = datasets
        self.allDatasets = [] 
        self.allFiles = []
        self.purpose = purpose
        for i in range(3):
            if len(self.purpose) < 3:
                self.purpose.append("")
        self.shiftDegree = shiftDegree
        self.information = info
        self.which = whichDatasets
        if self.purpose[0] == "data" or self.purpose[0] == "tilt" or self.purpose[0] == "if_sh":
            self.getData()
        elif self.purpose[0] == "figure" and self.purpose[1] == "comparison":
            self.fig4()
        elif self.purpose[0] == "figure" and self.purpose[1] == "if":
            self.fig5()
        elif self.purpose[0] == "figure" and self.purpose[1] == "flux":
            self.fig6()
        elif self.purpose[0] == "figure" and self.purpose[1] == "tilt":
            self.fig8()
        elif self.purpose[0] == "figure" and self.purpose[1] == "show":
            self.openFig()
        elif self.purpose[0] == "print":
            self.printCSV()
        elif self.purpose[0] == "tif":
            self.tif()
        elif self.purpose[0] == "stats":
            self.stats()
    def getData(self):
        for i in range(len(self.datasets)):
            if self.which == "All":
                print(self.datasets[i], "in whichDatasets")
                x = data(self.directory,self.datasetsNSA[i], self.shiftDegree, self.purpose) 
            elif self.datasets[i] in self.which:
                print(self.datasets[i], "in whichDatasets")
                x = data(self.directory,self.datasetsNSA[i], self.shiftDegree, self.purpose) 
            else:
                print(self.datasets[i], "not in whichDatasets")  
    def fig4(self):
        x = ComparisonToRoman(self.directory, self.datasetsNSA, self.shiftDegree)
        if self.information == 0:
            figure = x.aComparison()
            self.saveFig(figure, "aComparison", 1)
        if self.information == 1:
            figure = x.bComparison()
            self.saveFig(figure, "bComparison", 1)
        if self.information == 2:
            figure = x.cComparison()
            self.saveFig(figure, "cComparison", 1)
        if self.information >= 3:
            figure = x.dComparison()
            self.saveFig(figure, "dComparison", 1)
    def fig5(self):
        x = if_sh_Figure(self.directory, self.datasetsNSA, self.shiftDegree)
        if self.information == 0:
            figure = x.aIF(title = "Seasonal Evolution of Titan's Atmospheric Meridional Brightness Profile", xLabel = "Latitude", yLabel = "I/F", bands = [24,35,50], size = [16,16], cMap = "viridis", axisFontSize = 10, titleFontSize = 15, legendLocation = 4, legendFontSize = 6, lineWidth = 1, dataPointStyle = ".", lineStyles = ["solid", "solid", "solid"], grid = 1, cmapMin = 0, cmapMax = 1)
            self.saveFig(figure, "aIF", 2)
        elif self.information == 1:
            figure = x.bIF(title = "Seasonal Evolution of Titan's Atmospheric Meridional Brightness Profile", xLabel = "Latitude", yLabel = "I/F", bands = [24,35,50], size = [16,16], cMap = "viridis", axisFontSize = 10, titleFontSize = 15, legendLocation = 4, legendFontSize = 6, lineWidth = 1, dataPointStyle = ".", lineStyles = ["solid", "solid", "solid"], grid = 1, cmapMin = 0, cmapMax = 1)
            self.saveFig(figure, "bIF", 2)
        elif self.information == 2:
            figure = x.cIF()
            self.saveFig(figure, "cIF", 2)
        elif self.information == 3:
            figure = x.dIF()
            self.saveFig(figure, "dIF", 2)
        elif self.information == 4:
            figure = x.eIF()
            self.saveFig(figure, "eIF", 2)    
        elif self.information == 5:
            figure = x.fIF()
            self.saveFig(figure, "fIF", 2)
        elif self.information >= 6:
            figure = x.gIF()
            self.saveFig(figure, "gIF", 2)  
    def fig6(self):
        x = NS_Flux_Ratio(self.directory, self.datasetsNSA, self.shiftDegree)
        if self.information == 0:
            figure = x.aNS_Flux()
            self.saveFig(figure, "aNS_Flux", 3)
        elif self.information == 1:
            figure = x.bNS_Flux()
            self.saveFig(figure, "bNS_Flux", 3)
        elif self.information == 2:
            figure = x.cNS_Flux()
            self.saveFig(figure, "cNS_Flux", 3)
        elif self.information == 3:
            figure = x.dNS_Flux()
            self.saveFig(figure, "dNS_Flux", 3)
        elif self.information == 4:
            figure = x.eNS_Flux()
            self.saveFig(figure, "eNS_Flux", 3)
        elif self.information == 5:
            figure = x.fNS_Flux()
            self.saveFig(figure, "fNS_Flux", 3)
        elif self.information >= 6:
            figure = x.gNS_Flux()
            self.saveFig(figure, "gNS_Flux", 3)
    def fig8(self):
        x = Tilt(self.directory, self.datasetsNSA, self.shiftDegree)
        if self.information == 0:
            figure = x.aTiltPlot()
            self.saveFig(figure, "aTiltPlot", 4)
        elif self.information == 1:
            figure = x.bTiltPlot()
            self.saveFig(figure, "bTiltPlot", 4)
        elif self.information == 2:
            figure = x.cTiltPlot()
            self.saveFig(figure, "cTiltPlot", 4)
        elif self.information == 3:
            figure = x.dTiltPlot()
            self.saveFig(figure, "dTiltPlot", 4)   
        elif self.information >= 4:
            figure = x.eTiltPlot()
            self.saveFig(figure, "eTiltPlot", 4)
    def printCSV(self):
        printDatasets(self.directory, self.datasets, self.purpose)
    def tif(self):
        tif()
    def saveFig(self, figure, name, figType):
        if input("saveFig? ") == "y":
            import datetime 
            import pytz
            if type(figure) == list:
                for i in range(len(figure)):
                    current_time = datetime.datetime.now(pytz.timezone('America/New_York')) 
                    filename = self.directory[0] + "/" + self.directory[8][0] + "/" + self.directory[8][figType] +  "/" + name + "_" + str(i) + "_" + (str(current_time.year) + "-" + str(current_time.month) + "-" + str(current_time.day) + "-" + str(current_time.hour) + "-" + str(current_time.minute) + "-" + str(current_time.second))
                    print(filename)
                    figure[i].savefig((filename + ".svg"))
                    figure[i].savefig((filename + ".png"))
            else:  
                current_time = datetime.datetime.now(pytz.timezone('America/New_York')) 
                filename = self.directory[0] + "/" + self.directory[8][0] + "/" + self.directory[8][figType] +  "/" + name + "_" + (str(current_time.year) + "-" + str(current_time.month) + "-" + str(current_time.day) + "-" + str(current_time.hour) + "-" + str(current_time.minute) + "-" + str(current_time.second))
                print((filename + ".svg"))
                print((filename + ".png"))
                figure.savefig((filename + ".svg"))
                figure.savefig((filename + ".png"))
    def openFig(self):
        import glob
        import os
        import webbrowser
        folders = [self.directory[0] + "/" + self.directory[8][0] + "/" + x + "/" for x in self.directory[8]][1:5]
        flowchartFolder = [self.directory[0] + "/" + self.directory[8][0] + "/" + x + "/*" for x in self.directory[8]][5]
        inputx = input("type? ")
        if ".svg" not in inputx and ".png" not in inputx:
            self.openFig()
        try:
            listFiles = [x + '*' + inputx for x in folders]
            list_of_files = glob.glob(listFiles[0])
        except:
            print("try again")
            self.openFig()
        for folder in range(len(folders)):
            list_of_files = glob.glob(listFiles[folder])
            try:
                if len(list_of_files) == 0:
                    print("no file in", folder)
                else:
                    if folder == len(folders)-1:
                        latest_file = (max(list_of_files, key=os.path.getctime))
                        latestSplit = latest_file.split('-')
                        for files in list_of_files:
                            fileSplit = files.split('-')
                            if fileSplit[1:5] == latestSplit[1:5] and (fileSplit[0])[::5] == (latestSplit[0])[::5]:
                                files = os.path.abspath(files)
                                print(files)
                                if platform.system() == 'Darwin':       # macOS
                                    subprocess.call(('open', files))
                                elif platform.system() == 'Windows':    # Windows
                                    if inputx == ".svg":
                                        webbrowser.open_new_tab(files) 
                                    else:
                                        os.startfile(files)
                                else:                                   # linux variants
                                    subprocess.call(('xdg-open', files))
                    else:
                        latest_file = (max(list_of_files, key=os.path.getctime))
                        latest_file = os.path.abspath(latest_file)
                        print(latest_file)
                        if platform.system() == 'Darwin':       # macOS
                            subprocess.call(('open', latest_file))
                        elif platform.system() == 'Windows':    # Windows
                            if inputx == ".svg":
                                webbrowser.open_new_tab(latest_file) 
                            else:
                                os.startfile(latest_file)
                        else:                                   # linux variants
                            subprocess.call(('xdg-open', latest_file))
            except:
                print("error on folder", folders[folder])
                pass
        lists = glob.glob(flowchartFolder)
        latest_file = (max(lists, key=os.path.getctime))
        latest_file = os.path.abspath(latest_file)
        print(latest_file)
        try:
            if platform.system() == 'Darwin':       # macOS
                subprocess.call(('open', latest_file))
            elif platform.system() == 'Windows':    # Windows
                webbrowser.open_new_tab(latest_file) 
            else:                                   # linux variants
                subprocess.call(('xdg-open', latest_file))
        except:
            print("error on folder", folders[folder])
            pass
    def stats(self):
        folder = [_ for _ in os.listdir((self.directory[0])[0:-4] + '/Code/NS/') if _.endswith('.py')]
        folder = [(self.directory[0])[0:-4] + '/Code/NS/' + x for x in folder ]
        sum = 0
        methodSum = 0
        characterCount = 0
        fileCount = len(folder)
        for i in folder:
            with open(i) as file:
                x = 0
                for line in file:
                    characterCount+=len(line)
                    if "def " in line and "(self" in line:
                        methodSum+=1
                    x+=1
                sum+=x
        print("files:", str(fileCount), "\nfunctions in all files:", str(methodSum), "\nlines in all files:", str(sum), '\ncharacters in all files:', str(characterCount)) 
#input purpose here. List must be length 3.
#Titan(purpose = ["figure", "comparison"], info = 0, whichDatasets = "all")
#Titan(purpose = ["figure", "if"], info = 0, whichDatasets = "all")
#Titan(purpose = ["figure", "flux"], info = 0, whichDatasets = "all")
#Titan(purpose = ["figure", "tilt"], info = 0, whichDatasets = "all")]
if __name__ == "__main__":
    x = Titan(purpose = ["figure","if"], info = 100, whichDatasets = "All")
from data_process import data
import subprocess, os, platform
class Titan:
    def __init__(self, directory = {"flyby_parent_directory" : '/Users/aadvik/Desktop/NASA/North_South_Asymmetry/data', "wavelength_data": "titan_data/wavelength.csv", "flyby_data" : 'csv/', "flyby_image_directory" : 'vis.cyl', "flyby_info" : 'titan_data/flyby_data.csv', "if_scalars" : 'titan_data/IFData.csv', "analysis": '_analytics.csv', "analysis_folder" : 'Result/', "Figure names" : {"if" : 'Figures/IF', "flux" : 'Figures/NS_Flux_Ratio', "tilt" : 'Figures/Tilt', "flowchart" : 'Figures/Flowchart', "nsb" : 'Figures/North_South_Boundary', "figures"  : "Figures/"}, "lorenz_figure" : "titan_data/lorenz_2004/x.csv"},
                    shiftDegree = 6,
                    flybys =  [['Ta', -12, 15, [110,290]],['T8', -12, 15, [120,300]],['T31', -12, 15, [60,240]],
                                ['T61', -12, 15,[-120,-300]],['T62', -12, 15,  [79,247]],['T67', -12, 15, [-120,-300]],
                                ['T79', -10, 15, [60,240]],['T85', -10, 15, [60,240]],['T92', -5, 15, [100,280]],
                                ['T108', 0, 15, [100,280]],['T114', 0, 20, [60,240]],['278TI', 10, 25, [60,240]],
                                ['283TI', 10, 25, [60,240]]],
                    purpose = ["if_sh", [71,72,73], "show"],
                    whichDatasets = True,  info = []):
        self.directory = directory
        self.flyby_names = [flyby[0] for flyby in flybys]
        self.flybydata = flybys
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
    
    def getData(self):
        for i in range(len(self.flyby_names)):
            if self.which == "All" or self.which == "all":
                print(self.flyby_names[i], "in whichDatasets")
                x = data(self.directory,self.flybydata[i], self.shiftDegree, self.purpose) 
            elif self.flyby_names[i] in self.which:
                print(self.flyby_names[i], "in whichDatasets")
                x = data(self.directory,self.flybydata[i], self.shiftDegree, self.purpose) 
            else:
                print(self.flyby_names[i], "not in whichDatasets")  
    def saveFig(self, figure, name, figType):
        if input("saveFig? ") == "y":
            import datetime 
            import pytz
            if type(figure) == list:
                for i in range(len(figure)):
                    current_time = datetime.datetime.now(pytz.timezone('America/New_York')) 
                    filename = os.path.join(self.directory["flyby_parent_directory"], self.directory["Figure names"][figType], name + "_" + str(i) + "_" + (str(current_time.year) + "-" + str(current_time.month) + "-" + str(current_time.day) + "-" + str(current_time.hour) + "-" + str(current_time.minute) + "-" + str(current_time.second))).replace("\\","/")
                    print(filename)
                    figure[i].savefig((filename + ".svg"))
                    figure[i].savefig((filename + ".png"))
            else:  
                current_time = datetime.datetime.now(pytz.timezone('America/New_York')) 
                filename = os.path.join(self.directory["flyby_parent_directory"], self.directory["Figure names"][figType], name + "_" + (str(current_time.year) + "-" + str(current_time.month) + "-" + str(current_time.day) + "-" + str(current_time.hour) + "-" + str(current_time.minute) + "-" + str(current_time.second))).replace("\\","/")
                print((filename + ".svg"))
                print((filename + ".png"))
                figure.savefig((filename + ".svg"))
                figure.savefig((filename + ".png"))
    def openFig(self):
        import glob
        import os
        import webbrowser
        folders = [os.path.join(self.directory["flyby_parent_directory"], self.directory["Figure names"]["figures"], x).replace("\\", "/") for x in os.listdir(os.path.join(self.directory["flyby_parent_directory"],self.directory["Figure names"]["figures"]).replace("\\", "/"))]
        inputx = input("type? ")
        if ".svg" not in inputx and ".png" not in inputx:
            self.openFig()
        try:
            listFiles = [os.path.join(x,'*' + inputx).replace("\\","/") for x in folders]
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
    def stats(self):
        folder = [_ for _ in os.listdir(os.path.join('/'.join((self.directory["flyby_parent_directory"].replace("\\","/").split("/")[0:-1])), 'Code' , 'NS').replace("\\", "/")) if _.endswith('.py')]
        folder = [os.path.join('/'.join((self.directory["flyby_parent_directory"].replace("\\","/").split("/")[0:-1])), 'Code' , 'NS', x).replace("\\", "/") for x in folder ]
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
if __name__ == "__main__":
    x = Titan(purpose = ["data", "show"], info = 100, whichDatasets = "All")
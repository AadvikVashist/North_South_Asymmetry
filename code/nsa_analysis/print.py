import numpy as np
import pandas as pd
class printDatasets:
    def __init__(self, directory,Tdatasets,print):
        if print[1] == "all":
            self.whichFlyby = True
        else:
            self.whichFlyby = print[1]
        if print[2] == "all":
            self.row = True
        else:
            self.row = print[2]
        self.directory = directory
        self.Tdatasets = Tdatasets
        self.file = []
        self.mainPrint()
    def mainPrint(self):
        for i in self.Tdatasets:
            if self.whichFlyby == True or i in self.whichFlyby:
                file = (self.directory[0] + "/" + self.directory[7] + "/" + i + self.directory[5])
                self.print(file)
                input("input anything for next dataset")
    def print(self, file):
        print(file, "\n\n\n\n")
        self.file.append(np.array(pd.read_csv(file, header = None)))
        for i in range(len(self.file[-1])):
            if self.row == True or i in self.row:
                for x in self.file[-1][i]:
                    print(x)
                input("input anything for next filetype")
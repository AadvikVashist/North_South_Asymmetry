import numpy as np
import pandas as pd
import os
class printDatasets:
    def __init__(self, directory,Tdatasets,print):
        if type(print) == list:
            if print[0].lower() == "all":
                self.whichFlyby = True
            else:
                self.whichFlyby = print
        else:
            if print.lower() == "all":
                self.whichFlyby = True
            else:
                raise ValueError("print is wroong")
        self.directory = directory
        self.Tdatasets = Tdatasets
        self.file = []
        self.mainPrint()
    def mainPrint(self):
        for i in self.Tdatasets:
            if self.whichFlyby == True or i in self.whichFlyby:
                files = [os.path.join(self.directory["flyby_parent_directory"],i, self.directory["flyby_image_directory"], f).replace("\\","/")  for f in os.listdir(os.path.join(self.directory["flyby_parent_directory"],i, self.directory["flyby_image_directory"]).replace("\\","/"))]
                for file in files:
                    print(file)
        print("\n\n")
        
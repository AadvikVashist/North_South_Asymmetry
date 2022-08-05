import numpy as np
import pandas as pd
import time
class tim:
    def __init__(self):
        self.data = []; self.error = []; self.NSA = []
        tdata = ['Ta','T8','T31','T61','T62','T67','T79','T85','T92','T108','T114','278TI']
        for i in tdata:
            self.datasetRead('C:\\Users\\aadvi\\Desktop\\Titan Paper\\Data\\Result\\' + i + '_analytics.csv')
        for i in range(len(tdata)):
            for x in range(96):
                print(self.NSA[i][x])
            print("\n\n\n\n\n\n\n\n\n\n\n\n\n")
            for x in range(96):
                print(self.error[i][x])
            time.sleep(1)
            print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    def datasetRead(self,x):
            self.data = np.array(pd.read_csv(x, header = None))
            self.NSA.append(((self.data[0])[1:-1]).astype(np.float64))
            self.error.append(((self.data[1])[1:-1]).astype(np.float64))
tim()
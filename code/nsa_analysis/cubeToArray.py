import numpy as np
from pyvims.projections import equi_cube
from pyvims import VIMS
import pandas as pd
import matplotlib.pyplot as plt
""""
geo = pd.read_csv('C:/Users/aadvi/Desktop/Titan Paper/Data/T114/csv/t114_nsa_geo.csv', header=None)
geo = np.array(geo)[:,1]
x = geo.reshape(181,363)
plt.imshow(x)
plt.show()"""
class cubeToArray:
    def __init__(self):
        self.Lat = []
        self.Lon = []
        cubs = ["1875606024_1","1875604404_1","1875618780_1","1875618279_1","1875617102_1","1875615197_1",
        "1875611132_1","1875630978_1","1875631459_1","1875632672_1","1875633165_1","1875645437_1",
        "1875647049_1","1875647318_1","1875656672_1","1875658704_1"]
        csv_grid = np.zeros(shape=(96,360*720))
        dpi=80; min_if=[]; max_if=[]
        
        for wv in range(96):
            fin = np.zeros(shape=(720,360)); fin=fin.T
            plt.figure();plt.imshow(fin);plt.colorbar()
            plt.show()
            for idx in range(len(cubs)):
                c = VIMS(cubs[idx], channel='vis')
                img, (x, y), extent, cnt = equi_cube(c, wv+1, ppd=2)
                init = np.zeros(shape=(720,360))
                init = np.empty((720,360)); init[:] = np.NaN
                init = init.T
                print(init)
                ## create superimposed grids for lat, lon
                lat_ind = np.where(np.isin(np.repeat(np.arange(-89.75,89.75+0.5,0.5),720).reshape(-1,720), y))
                lon_ind = np.where(np.isin(np.repeat(np.arange(-179.75,179.75+0.5,0.5),360).reshape(-1,360).T , x))
                x_lon = lon_ind[1]
                y_lat = lat_ind[0]
                self.Lat.append(lat_ind)
                self.Lon.append(lon_ind)
                plt.imshow(np.array(lat_ind).reshape(720,360))
                plt.show()
                plt.imshow(np.array(lon_ind).reshape(720,360))
                plt.show()
                print(c.flyby)
x = cubeToArray()
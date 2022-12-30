from pyvims import VIMS
import numpy as np
import pandas as pd

data = np.array(pd.read_csv("/Users/aadvik/Desktop/NASA/North_South_Assymetry/data/flyby_info/nantes_cubes.csv"))
for cube in data:
    cube_data = VIMS(cube)
    if 95 in cube_data.bands:
        x = 0
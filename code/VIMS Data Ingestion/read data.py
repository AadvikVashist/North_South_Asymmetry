import numpy as np
import pandas as pd
nantes_csv = pd.read_csv('C:/Users/aadvi/Desktop/North_South_Asymmetry/data/flyby_info/nantes.csv')
nantes_csv = np.array(nantes_csv)
targeted_flybys = [row for row in list(nantes_csv) if "|" in row[1]]
non_targeted_flybys = [row for row in list(nantes_csv) if "|" not in row[1]]
nantes_csv = nantes_csv
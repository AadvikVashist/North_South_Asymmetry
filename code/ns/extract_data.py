import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def get_lorenz_data():
    csv =pd.read_csv("data/Titan Data/Lorenz_2004/x.csv")
    csv = csv = csv.to_dict()
    new_dict ={}
    for key in csv.keys():
        if "Unnamed" not in key:
            new_dict[key] = {"x" : list(csv[key].values())[1::]}
        else:
            key_send = list(new_dict.keys())[-1]
            new_dict[key_send]["y"] = list(csv[key].values())[1::]
    newt_dict = {}
    for key, value in new_dict.items():
        x = np.array(value["x"], dtype = np.float32)
        y = np.array(value["y"], dtype = np.float32)
        x = [a for a in x if not np.isnan(a)]
        y = [a for a in y if not np.isnan(a)]
        xy = zip(x,y)
        sorted_pairs = sorted((i,j) for i,j in zip(x,y))
        x,y = zip(*sorted_pairs)
        newt_dict[key] = [x,y]
    #     plt.plot(x,y)
    # plt.show()
    return newt_dict

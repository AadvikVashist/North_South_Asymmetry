import numpy as np
import pandas as pd
import pickle
def get_limb(csv):
    header = csv[0]
    header_index = [index for index,h in enumerate(header) if "limb vis" in h.lower()][0]
    return [header] + [c for c in csv if "yes" in c[header_index].lower()]
def get_percentage(csv,min):
    header = csv[0]
    header_index = [index for index,h in enumerate(header) if "percentage" in h.lower()][0]
    return [header] + [c for c in csv[1::] if float(c[header_index]) >= min]
def get_vis(csv):
    header = csv[0]
    header_index = [index for index,h in enumerate(header) if "sampling mode" in h.lower()][0]
    header_split_index = [index for index,h in enumerate(header[header_index].split("|")) if "vis" in h.lower()][0]
    return [header] + [c for c in csv[1::] if "n/a" not in c[header_index].lower().split("|")[header_split_index]]
def get_ir(csv):
    header = csv[0]
    header_index = [index for index,h in enumerate(header) if "sampling mode" in h.lower()][0]
    header_split_index = [index for index,h in enumerate(header[header_index].split("|")) if "ir" in h.lower()][0]
    return [header] + [c for c in csv[1::] if "n/a" not in c[header_index].lower().split("|")[header_split_index]]
def get_filtered_phase(csv,phase_list : list):
    header = csv[0]
    header_index = [index for index,h in enumerate(header) if "phase" in h.lower()][0]
    return [header] + [c for c in csv[1::] if float(c[header_index]) >= min(phase_list) and float(c[header_index]) <= max(phase_list)]

if __name__ == "__main__":
    with open("code/VIMS Data Ingestion/data/nantes_cubes.pickle", "rb") as f:
        # Use pickle to dump the variable into the file
        data = pickle.load(f)    

    limb = get_limb(data)
    limb = get_percentage(limb, 0.5)
    limb = get_filtered_phase(limb,[0,40])
    print(limb)